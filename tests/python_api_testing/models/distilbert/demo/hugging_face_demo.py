import sys
import torch
from pathlib import Path
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../torch")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from utils import get_answer
from transformers import DistilBertForQuestionAnswering, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
HF_model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)

question, context = (
    "Where do I live?",
    "My name is Merve and I live in İstanbul.",
)
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    torch_output = HF_model(**inputs)

torch_answer = get_answer(inputs, torch_output, tokenizer)

logger.info("HF Model answered")
logger.info(torch_answer)
