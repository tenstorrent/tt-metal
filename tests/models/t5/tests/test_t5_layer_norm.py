# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from loguru import logger
import pytest
from transformers import T5Model

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import comp_pcc, comp_allclose
from models.t5.tt.t5_layer_norm import TtT5LayerNorm


def run_test_T5LayerNorm_inference(pcc, device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = hf_reference_model.config

    # Module to test
    if config.is_decoder:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[1].layer_norm
        base_address = f"decoder.block.0.layer.1.layer_norm"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1].layer_norm
        base_address = f"encoder.block.0.layer.1.layer_norm"

    # Prepare input
    t5_layer_norm_input = (torch.rand(1, 1, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(t5_layer_norm_input)[0].unsqueeze(1)
    tt_T5LayerNorm_model = TtT5LayerNorm(
        config, hf_reference_model.state_dict(), base_address, device
    )

    # TT hardware execution
    tt_layer_norm_input = torch_to_tt_tensor_rm(t5_layer_norm_input, device)

    tt_out = tt_T5LayerNorm_model(tt_layer_norm_input)
    tt_out = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5LayerNorm_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5LayerNorm_inference {model_name} Failed!")

    assert does_pass, f"T5LayerNorm output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 64, 512),),
)
def test_T5LayerNorm_inference_t5_small(pcc, model_name, input_h, input_w, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5LayerNorm_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "google/flan-t5-small", 64, 512),),
)
def test_T5LayerNorm_inference_flan_t5_small(
    pcc, model_name, input_h, input_w, reset_seeds
):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5LayerNorm_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-base", 64, 768),),
)
def test_T5LayerNorm_inference_t5_base(pcc, model_name, input_h, input_w, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5LayerNorm_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)
