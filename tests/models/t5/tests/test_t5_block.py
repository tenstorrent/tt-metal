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
from models.t5.tt.t5_block import TtT5Block


def run_test_T5Block_inference(pcc, device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = hf_reference_model.config
    config.is_decoder = False

    block = 1
    has_relative_attention_bias = block == 0

    if config.is_decoder:
        hf_reference_module = hf_reference_model.decoder.block[block]
        base_address = f"decoder.block.{block}"
    else:
        hf_reference_module = hf_reference_model.encoder.block[block]
        base_address = f"encoder.block.{block}"

    # Prepare input
    test_input = (torch.rand(32, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(0)
    test_input = test_input.unsqueeze(0)

    tt_model = TtT5Block(
        config,
        hf_reference_model.state_dict(),
        base_address,
        device,
        has_relative_attention_bias,
    )
    tt_out = tt_model(torch_to_tt_tensor_rm(test_input, device, put_on_device=True))[0]
    tt_out = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5Block_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Block_inference {model_name} Failed!")

    assert does_pass, f"T5Block output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 64, 512),),
)
def test_T5Block_inference_t5_small(pcc, model_name, input_h, input_w, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5Block_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "google/flan-t5-small", 64, 512),),
)
def test_T5Block_inference_flan_t5_small(
    pcc, model_name, input_h, input_w, reset_seeds
):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5Block_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-base", 64, 768),),
)
def test_T5Block_inference_t5_base(pcc, model_name, input_h, input_w, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5Block_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)
