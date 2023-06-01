from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from transformers import BloomForCausalLM, BloomTokenizerFast
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.bloom.bloom_causal_lm as bloom_causal_lm


def test_bloom_causal_lm():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m", torchscript=False
    )
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_causal_lm = bloom_causal_lm.TtBloomForCausalLM(config, state_dict, device)
    pt_bloom_causal_lm = hugging_bloom_reference_model

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = tokenized.input_ids

    pt_out = pt_bloom_causal_lm.forward(input_ids)
    tt_out = tt_bloom_causal_lm.forward(device, input_ids)

    pt_out = pt_out[0]
    tt_out = tt_out[0]

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_causal_lm: Passed!")
    else:
        logger.warning("bloom_causal_lm: Failed!")

    assert does_pass
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_causal_lm()
