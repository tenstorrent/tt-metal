# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.transfuser.tests.reference.self_attention import SelfAttention

# from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


# def create_gpt_preprocessor(device, weight_dtype=ttnn.bfloat16):
#     def custom_preprocessor(torch_model, name, ttnn_module_args):
#         parameters = {}
#         if hasattr(torch_model, "fc1") and hasattr(torch_model, "fc2"):  # MLP model
#             parameters["fc1"] = {}
#             parameters["fc2"] = {}

#             # Preprocess fc1 layer parameters
#             parameters["fc1"]["weight"] = preprocess_linear_weight(torch_model.fc1.weight, dtype=weight_dtype)
#             parameters["fc1"]["bias"] = preprocess_linear_bias(torch_model.fc1.bias, dtype=weight_dtype)

#             # Preprocess fc2 layer parameters
#             parameters["fc2"]["weight"] = preprocess_linear_weight(torch_model.fc2.weight, dtype=weight_dtype)
#             parameters["fc2"]["bias"] = preprocess_linear_bias(torch_model.fc2.bias, dtype=weight_dtype)

#         return parameters

#     return custom_preprocessor


@pytest.mark.parametrize(
    "n_embed, n_head, attn_pdrop, resid_pdrop, input_shape",
    ((72, 4, 0.1, 0.1, (1, 174, 72)),),  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b])
def test_self_attn(device, n_embed, n_head, attn_pdrop, resid_pdrop, input_shape, input_dtype, weight_dtype):
    x = torch.randn(input_shape)

    ref_layer = SelfAttention(
        n_embd=n_embed,
        n_head=n_head,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
    )

    ref_output = ref_layer(x)
    print(ref_output)
    pytest.skip("Skipping self attention test")
    # parameters = preprocess_model_parameters(
    #     initialize_model=lambda: ref_layer,
    #     custom_preprocessor=create_mlp_preprocessor(device, weight_dtype),
    #     device=device,
    # )

    # TODO: Implement self attention
    tt_layer = TTSelfAttention()
    tt_input = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype)
    tt_input = ttnn.to_memory_config(tt_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)

    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.99)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("SelfAttention Passed!")
    else:
        logger.warning("SelfAttention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
