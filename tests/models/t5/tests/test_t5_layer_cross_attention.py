# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from loguru import logger
import pytest
from transformers import T5Model
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.utility_functions import comp_pcc, comp_allclose
from models.t5.tt.t5_layer_cross_attention import TtT5LayerCrossAttention


def run_test_T5LayerCrossAttention_inference(pcc, device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = hf_reference_model.config
    config.is_decoder = False

    # Cross attention can be only decoder
    hf_reference_module = hf_reference_model.decoder.block[0].layer[1]
    base_address = f"decoder.block.0.layer.1"

    # Cross attention is only in decoder part
    config.is_decoder = True

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(32, input_h, input_w) * 2) - 1
    key_value_states = (torch.rand(32, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input, key_value_states)[0].unsqueeze(0)

    test_input = test_input.unsqueeze(0)
    key_value_states = key_value_states.unsqueeze(0)

    tt_model = TtT5LayerCrossAttention(
        config, hf_reference_model.state_dict(), base_address, device
    )
    tt_out = tt_model(
        torch_to_tt_tensor_rm(test_input, device, put_on_device=True),
        torch_to_tt_tensor_rm(key_value_states, device, put_on_device=True),
    )[0]
    tt_out = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5LayerCrossAttention_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5LayerCrossAttention_inference {model_name} Failed!")

    assert (
        does_pass
    ), f"T5LayerCrossAttention output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 64, 512),),
)
def test_T5LayerCrossAttention_inference_t5_small(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5LayerCrossAttention_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "google/flan-t5-small", 64, 512),),
)
def test_T5LayerCrossAttention_inference_flan_t5_small(
    pcc, model_name, input_h, input_w
):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5LayerCrossAttention_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-base", 64, 768),),
)
def test_T5LayerCrossAttention_inference_t5_base(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5LayerCrossAttention_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)
