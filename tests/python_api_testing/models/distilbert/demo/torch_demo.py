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

from modeling_distilbert import PytorchDistilBertForQuestionAnswering
from utils import get_answer

from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)

state_dict = HF_model.state_dict()
config = HF_model.config
get_head_mask = HF_model.distilbert.get_head_mask

PT_model = PytorchDistilBertForQuestionAnswering(config)
res = PT_model.load_state_dict(state_dict)
PT_model.distilbert.get_head_mask = get_head_mask

question, context = (
    "Where do I live?",
    "My name is Merve and I live in İstanbul.",
)

inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    pt_output = PT_model(**inputs)

pt_answer = get_answer(inputs, pt_output, tokenizer)

logger.info("PT Model answered")
logger.info(pt_answer)
