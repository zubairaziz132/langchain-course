from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langsmith import Client

load_dotenv()



def main():
    information="""
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada;
 he had obtained Canadian citizenship at birth through his Canadian-born mother.
  He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, 
  United States, before moving to California to pursue business ventures. In 1995, 
  Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com,
   an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.
                """
    summary_template = """
Given the information {information}, about a person I want to create:
1. a short Summary
2. a short PersonSummary
3. two facts about person
"""

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    #llm = ChatOllama(model="llama2")
    llm = ChatOllama(temperature=0,model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
