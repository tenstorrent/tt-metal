# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest
from transformers import T5Model
from loguru import logger
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.utility_functions import comp_pcc, comp_allclose
from models.t5.tt.t5_attention import TtT5Attention


def run_test_T5Attention_inference(
    pcc, device, block, use_mask, model_name, input_h, input_w
):
    hugging_face_reference_model = T5Model.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # Input is (batch_size, seq_length, dim)
    test_input = ((torch.rand(1, input_h, input_w) * 2) - 1) / 512

    mask = -65504.0 * torch.cat([torch.zeros(7), torch.ones(25)]) / 2
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mask = mask if use_mask else None

    config = hugging_face_reference_model.config
    config.is_decoder = False
    has_relative_attention_bias = bool(block == 0)

    # Module to test
    if config.is_decoder:
        hf_reference_module = (
            hugging_face_reference_model.decoder.block[block].layer[0].SelfAttention
        )
        base_address = f"decoder.block.{block}.layer.0.SelfAttention"
    else:
        hf_reference_module = (
            hugging_face_reference_model.encoder.block[block].layer[0].SelfAttention
        )
        base_address = f"encoder.block.{block}.layer.0.SelfAttention"

    pytorch_model = hf_reference_module
    pt_out = pytorch_model(hidden_states=test_input, mask=mask)[0].unsqueeze(0)

    test_input = test_input.unsqueeze(0)
    tt_test_input = torch_to_tt_tensor_rm(test_input, device, put_on_device=True)

    tt_model = TtT5Attention(
        config,
        hugging_face_reference_model.state_dict(),
        base_address,
        device,
        has_relative_attention_bias,
    )
    tt_out = tt_model(hidden_states=tt_test_input, mask=mask)[0]
    tt_out = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5Attention_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Attention_inference {model_name} Failed!")

    assert does_pass, f"T5Attention output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 64, 512),),
)
def test_T5Attention_block_0_no_mask_t5_small(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5Attention_inference(
        pcc,
        device,
        block=0,
        use_mask=False,
        model_name=model_name,
        input_h=input_h,
        input_w=input_w,
    )
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 32, 512),),
)
def test_T5Attention_block_2_no_mask_t5_small(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Attention_inference(
        pcc,
        device,
        block=2,
        use_mask=False,
        model_name=model_name,
        input_h=input_h,
        input_w=input_w,
    )
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.98, "t5-small", 32, 512),),
)
def test_T5Attention_block_0_with_mask_t5_small(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Attention_inference(
        pcc,
        device,
        block=0,
        use_mask=True,
        model_name=model_name,
        input_h=input_h,
        input_w=input_w,
    )
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "google/flan-t5-small", 32, 512),),
)
def test_T5Attention_block_0_no_mask_flan_t5_small(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5Attention_inference(
        pcc,
        device,
        block=0,
        use_mask=False,
        model_name=model_name,
        input_h=input_h,
        input_w=input_w,
    )
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-base", 32, 768),),
)
def test_T5Attention_block_0_no_mask_t5_base(pcc, model_name, input_h, input_w):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Attention_inference(
        pcc,
        device,
        block=0,
        use_mask=False,
        model_name=model_name,
        input_h=input_h,
        input_w=input_w,
    )
    tt_lib.device.CloseDevice(device)
