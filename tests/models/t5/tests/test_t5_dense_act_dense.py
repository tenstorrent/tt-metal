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
from models.t5.tt.t5_dense_act_dense import TtT5DenseActDense


def run_test_T5DenseActDense_inference(pcc, device, model_name, input_h, input_w):
    hugging_face_reference_model = T5Model.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    config = hugging_face_reference_model.config
    config.is_decoder = False

    if config.is_decoder:
        hf_reference_module = (
            hugging_face_reference_model.decoder.block[0].layer[2].DenseReluDense
        )
        base_address = f"decoder.block.0.layer.2.DenseReluDense"
    else:
        hf_reference_module = (
            hugging_face_reference_model.encoder.block[0].layer[1].DenseReluDense
        )
        base_address = f"encoder.block.0.layer.1.DenseReluDense"

    # Prepare input
    test_input = (torch.rand(1, 1, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    tt_model = TtT5DenseActDense(
        config, hugging_face_reference_model.state_dict(), base_address, device
    )
    tt_out = tt_model(torch_to_tt_tensor_rm(test_input, device, put_on_device=True))
    tt_out = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5DenseActDense_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5DenseActDense_inference {model_name} Failed!")

    assert does_pass, f"T5DenseActDense output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-small", 64, 512),),
)
def test_T5DenseActDense_inference_t5_small(
    pcc, model_name, input_h, input_w, reset_seeds
):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5DenseActDense_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name ,input_h, input_w",
    ((0.99, "t5-base", 64, 768),),
)
def test_T5DenseActDense_inference_t5_base(
    pcc, model_name, input_h, input_w, reset_seeds
):
    device = tt_lib.device.CreateDevice(0)
    run_test_T5DenseActDense_inference(pcc, device, model_name, input_h, input_w)
    tt_lib.device.CloseDevice(device)
