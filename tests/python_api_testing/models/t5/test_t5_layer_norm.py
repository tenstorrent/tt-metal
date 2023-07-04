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
from python_api_testing.models.utility_functions_new import (
    tt2torch_tensor,
    torch2tt_tensor,
)
from python_api_testing.models.t5.t5_layer_norm import TtT5LayerNorm


def run_test_T5LayerNorm_inference(device, model_name):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    # Module to test
    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[1].layer_norm
        base_address = f"decoder.block.0.layer.1.layer_norm"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1].layer_norm
        base_address = f"encoder.block.0.layer.1.layer_norm"

    # Prepare input
    torch.manual_seed(0)
    t5_layer_norm_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(t5_layer_norm_input)[0].unsqueeze(1)
    tt_T5LayerNorm_model = TtT5LayerNorm(
        config, hf_reference_model.state_dict(), base_address, device
    )

    # TT hardware execution
    tt_layer_norm_input = torch2tt_tensor(t5_layer_norm_input, device)

    tt_out = tt_T5LayerNorm_model(tt_layer_norm_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_T5LayerNorm_inference Passed!")
    else:
        logger.warning("test_T5LayerNorm_inference Failed!")

    assert does_pass


def test_T5LayerNorm_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5LayerNorm_inference(device, "t5-small")
    tt_lib.device.CloseDevice(device)


def test_T5LayerNorm_inference_flan():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5LayerNorm_inference(device, "google/flan-t5-small")
    tt_lib.device.CloseDevice(device)
