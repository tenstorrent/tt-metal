# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest
from loguru import logger

from transformers import T5Model
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.utility_functions import comp_pcc, comp_allclose
from models.t5.tt.t5_dense_gated_act_dense import TtT5DenseGatedActDense


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_T5DenseGatedActDense_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    hugging_face_reference_model = T5Model.from_pretrained("google/flan-t5-small")
    hugging_face_reference_model.eval()

    config = hugging_face_reference_model.config

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
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    tt_model = TtT5DenseGatedActDense(
        config, hugging_face_reference_model.state_dict(), base_address, device
    )
    tt_out = tt_model(torch_to_tt_tensor_rm(test_input, device, put_on_device=True))
    tt_out = tt_to_torch_tensor(tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)
    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("test_T5DenseGatedActDense_inference Passed!")
    else:
        logger.warning("test_T5DenseGatedActDense_inference Failed!")

    assert (
        does_pass
    ), f"T5DenseGatedActDense output does not meet PCC requirement {pcc}."
