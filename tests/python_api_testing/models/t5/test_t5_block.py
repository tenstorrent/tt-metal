from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import json
import tt_lib
from loguru import logger

from transformers import T5Model
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.t5.t5_block import TtT5Block


def run_test_T5Block_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    block = 1
    has_relative_attention_bias = block == 0

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[block]
        base_address = f"decoder.block.{block}"
    else:
        hf_reference_module = hf_reference_model.encoder.block[block]
        base_address = f"encoder.block.{block}"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(32, 128, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(0)
    test_input = test_input.unsqueeze(0)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5Block(
        config,
        hf_reference_model.state_dict(),
        base_address,
        device,
        has_relative_attention_bias,
    )
    tt_out = tt_model(torch2tt_tensor(test_input, device))[0]
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5Block_inference Passed!")
    else:
        logger.warning("test_T5Block_inference Failed!")


def test_T5Block_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5Block_inference(device)
    tt_lib.device.CloseDevice(device)
