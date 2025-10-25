# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.transfuser.reference.gpt_block import Block
from models.experimental.transfuser.tt.gpt_block import TTGptBlock

from models.experimental.transfuser.tests.test_self_attention import create_self_attn_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_gpt_block_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "ln1") and hasattr(torch_model, "ln2"):
            parameters["ln1_weight"] = preprocess_linear_weight(torch_model.ln1.weight, dtype=weight_dtype)
            parameters["ln1_bias"] = preprocess_linear_weight(torch_model.ln1.bias, dtype=weight_dtype)
            parameters["ln2_weight"] = preprocess_linear_weight(torch_model.ln2.weight, dtype=weight_dtype)
            parameters["ln2_bias"] = preprocess_linear_weight(torch_model.ln2.bias, dtype=weight_dtype)
        if hasattr(torch_model, "attn"):
            self_attn_params = preprocess_model_parameters(
                initialize_model=lambda: torch_model.attn,
                custom_preprocessor=create_self_attn_preprocessor(device, weight_dtype),
                device=device,
            )
            parameters["attn"] = self_attn_params
        if hasattr(torch_model, "mlp"):
            parameters["mlp_0_weight"] = preprocess_linear_weight(torch_model.mlp[0].weight, dtype=weight_dtype)
            parameters["mlp_0_bias"] = preprocess_linear_weight(torch_model.mlp[0].bias, dtype=weight_dtype)
            parameters["mlp_2_weight"] = preprocess_linear_weight(torch_model.mlp[2].weight, dtype=weight_dtype)
            parameters["mlp_2_bias"] = preprocess_linear_weight(torch_model.mlp[2].bias, dtype=weight_dtype)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "n_embed, n_head, block_exp, attn_pdrop, resid_pdrop, input_shape",
    ((72, 4, 4, 0.1, 0.1, (1, 174, 72)),),  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_gpt_block(device, n_embed, n_head, block_exp, attn_pdrop, resid_pdrop, input_shape, input_dtype, weight_dtype):
    x = torch.randn(input_shape)

    ref_layer = Block(
        n_embd=n_embed,
        n_head=n_head,
        block_exp=block_exp,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
    ).eval()
    ref_output = ref_layer(x)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_gpt_block_preprocessor(device, weight_dtype),
        device=device,
    )
    tt_layer = TTGptBlock(device, parameters, n_head, dtype=weight_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_input = ttnn.from_torch(
        x, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.95)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("GPT Block Passed!")
    else:
        logger.warning("GPT Block Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
