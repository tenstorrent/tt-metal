# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from loguru import logger

from transformers import T5Model
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.utility_functions import comp_pcc, comp_allclose
from models.t5.tt.t5_stack import TtT5Stack


def run_test_T5Stack_inference(device, model_name, input_h, input_w, pcc):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = hf_reference_model.config
    config.is_decoder = False
    config.use_cache = False

    if config.is_decoder:
        hf_reference_module = hf_reference_model.decoder
        base_address = f"decoder"
    else:
        hf_reference_module = hf_reference_model.encoder
        base_address = f"encoder"

    # Prepare input
    test_input = (torch.rand(2, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(inputs_embeds=test_input)
    pt_out = pt_out.last_hidden_state
    pt_out = pt_out.unsqueeze(0)

    # Move test input to Tt device test_input
    test_input = test_input.unsqueeze(0)
    test_input = torch_to_tt_tensor_rm(test_input, device, put_on_device=True)

    tt_model = TtT5Stack(config, hf_reference_model.state_dict(), base_address, device)
    tt_model_outputs = tt_model(inputs_embeds=test_input)
    last_hidden_state = tt_model_outputs.last_hidden_state
    tt_out = tt_to_torch_tensor(last_hidden_state)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5Stack_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Stack_inference {model_name} Failed!")

    assert does_pass, f"T5Stack output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 64, 512),),
)
def test_T5Stack_inference_t5_small(pcc, model_name, input_h, input_w, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Stack_inference(device, model_name, input_h, input_w, pcc)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.94, "google/flan-t5-small", 64, 512),),
)
def test_T5Stack_inference_flan_t5_small(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Stack_inference(device, model_name, input_h, input_w, pcc)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-base", 64, 768),),
)
def test_T5Stack_inference_t5_base(pcc, model_name, input_h, input_w, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Stack_inference(device, model_name, input_h, input_w, pcc)
    tt_lib.device.CloseDevice(device)
