# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.transfuser.reference.self_attention import SelfAttention
from models.experimental.transfuser.tt.self_attn import TTSelfAttention

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_self_attn_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if (
            hasattr(torch_model, "key")
            and hasattr(torch_model, "query")
            and hasattr(torch_model, "value")
            and hasattr(torch_model, "proj")
        ):  # MLP model
            parameters["key"] = {}
            parameters["query"] = {}
            parameters["value"] = {}
            parameters["proj"] = {}

            # Preprocess key layer parameters
            parameters["key"]["weight"] = preprocess_linear_weight(torch_model.key.weight, dtype=weight_dtype)
            parameters["key"]["bias"] = preprocess_linear_bias(torch_model.key.bias, dtype=weight_dtype)

            # Preprocess query layer parameters
            parameters["query"]["weight"] = preprocess_linear_weight(torch_model.query.weight, dtype=weight_dtype)
            parameters["query"]["bias"] = preprocess_linear_bias(torch_model.query.bias, dtype=weight_dtype)

            # Preprocess value layer parameters
            parameters["value"]["weight"] = preprocess_linear_weight(torch_model.value.weight, dtype=weight_dtype)
            parameters["value"]["bias"] = preprocess_linear_bias(torch_model.value.bias, dtype=weight_dtype)

            # Preprocess proj layer parameters
            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=weight_dtype)
            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=weight_dtype)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "n_embed, n_head, attn_pdrop, resid_pdrop, input_shape",
    ((72, 4, 0.1, 0.1, (1, 174, 72)),),  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_self_attn(device, n_embed, n_head, attn_pdrop, resid_pdrop, input_shape, input_dtype, weight_dtype):
    x = torch.randn(input_shape)

    ref_layer = SelfAttention(
        n_embd=n_embed,
        n_head=n_head,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
    ).eval()
    # import pdb; pdb.set_trace()
    print(ref_layer)
    ref_output = ref_layer(x)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_self_attn_preprocessor(device, weight_dtype),
        device=device,
    )
    tt_layer = TTSelfAttention(
        device, parameters, n_embed, n_head, dtype=weight_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_input = ttnn.from_torch(
        x, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("SelfAttention Passed!")
    else:
        logger.warning("SelfAttention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
