from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from transformers import (
    BloomForQuestionAnswering,
    AutoTokenizer,
    BloomTokenizerFast,
    pipeline,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.bloom_new.bloom_qa as bloom_qa


def run_bloom_qa_inference(device):
    torch.manual_seed(0)
    model_name = "bigscience/bloom-560m"
    hugging_bloom_reference_model = BloomForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_qa = bloom_qa.TtBloomForQuestionAnswering(config, state_dict, device)
    pt_bloom_qa = hugging_bloom_reference_model

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    pt_out = pt_bloom_qa.forward(
        input_ids=input_ids
    )  # , attention_mask=attention_mask)
    tt_out = tt_bloom_qa.forward(
        device, input_ids=input_ids
    )  # , attention_mask=attention_mask)

    pt_start_logits = pt_out[0]  # start_logits
    pt_end_logits = pt_out[1]

    tt_start_logits = tt_out[0]
    tt_start_logits = tt_start_logits.squeeze(0)

    tt_end_logits = tt_out[1]
    tt_end_logits = tt_end_logits.squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_start_logits, tt_start_logits, 0.41)

    logger.info(comp_allclose(pt_start_logits, tt_start_logits))
    logger.info(pcc_message)

    does_pass, pcc_message = comp_pcc(pt_end_logits, tt_end_logits, 0.41)

    logger.info(comp_allclose(pt_end_logits, tt_end_logits))
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_qa: Passed!")
    else:
        logger.warning("bloom_qa: Failed!")

    assert does_pass


def test_bloom_qa():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_bloom_qa_inference(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_qa()
