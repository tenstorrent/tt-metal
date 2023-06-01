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

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_model as bloom_model


def run_bloom_model_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m", torchscript=False
    )
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    # use_cache
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = "transformer"
    # hidden_size = config.hidden_size # 1024
    # n_head = config.n_head

    tt_bloom_model = bloom_model.TtBloomModel(config, state_dict, base_address, device)
    pt_bloom_model = hugging_bloom_reference_model.transformer

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = tokenized.input_ids

    pt_out = pt_bloom_model.forward(input_ids)[0]
    tt_out = tt_bloom_model.forward(device, input_ids)[0]

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    tt_out_converted = tt_out_converted.squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_model: Passed!")
    else:
        logger.warning("bloom_model: Failed!")

    assert does_pass


def test_bloom_model():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_bloom_model_test(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_model()
