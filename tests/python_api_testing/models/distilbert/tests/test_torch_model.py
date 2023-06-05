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
from utils import comp_outputs, get_answer

from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from transformers import AutoTokenizer


def test_distilbert_qa_inference():
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
        HF_output = HF_model(**inputs)
        torch_output = PT_model(**inputs)

    torch_answer = get_answer(inputs, HF_output, tokenizer)
    pt_answer = get_answer(inputs, torch_output, tokenizer)

    logger.info("HF Model answered")
    logger.info(torch_answer)

    logger.info("PT Model answered")
    logger.info(pt_answer)

    does_pass_1, does_pass_2 = comp_outputs(HF_output, torch_output)

    if does_pass_1 and does_pass_2:
        logger.info("distilbertForQA Passed!")
    else:
        logger.warning("distilbertForQA Failed!")

    assert does_pass_1 and does_pass_2


if __name__ == "__main__":
    test_distilbert_qa_inference()
