# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.transfuser.reference.self_attention import SelfAttention
from models.experimental.transfuser.tt.self_attn import TTSelfAttention

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_bias, preprocess_linear_weight
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_self_attn_preprocessor(device, weight_dtype=ttnn.bfloat16, use_optimized=True):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if (
            hasattr(torch_model, "key")
            and hasattr(torch_model, "query")
            and hasattr(torch_model, "value")
            and hasattr(torch_model, "proj")
        ):
            if use_optimized:
                # Optimized version: Fused QKV
                n_head = torch_model.n_head
                n_embed = torch_model.query.weight.shape[-1]
                head_size = n_embed // n_head
                padded_head_size = ((head_size + 31) // 32) * 32
                padded_embed_dim = padded_head_size * n_head

                def pad_linear_layer(linear_layer, target_out_features):
                    """Pads a linear layer's weight and bias to the target out_features dimension."""
                    weight = linear_layer.weight
                    bias = linear_layer.bias

                    out_features, in_features = weight.shape
                    pad_size = target_out_features - out_features
                    if pad_size > 0:
                        # Pad along output dimension
                        weight = torch.cat([weight, torch.zeros(pad_size, in_features)], dim=0)
                        bias = torch.cat([bias, torch.zeros(pad_size)], dim=0)
                    return weight, bias

                # ---- Pad each Q, K, V ----
                q_weight, q_bias = pad_linear_layer(torch_model.query, padded_embed_dim)
                k_weight, k_bias = pad_linear_layer(torch_model.key, padded_embed_dim)
                v_weight, v_bias = pad_linear_layer(torch_model.value, padded_embed_dim)

                # ---- Fuse QKV ----
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

                print(f"QKV weight shape: {qkv_weight.shape}, bias shape: {qkv_bias.shape}")

                parameters["query_key_value"] = {
                    "weight": preprocess_linear_weight(qkv_weight, dtype=weight_dtype),
                    "bias": preprocess_linear_bias(qkv_bias, dtype=weight_dtype),
                }
            else:
                # Old version: Separate Q, K, V (from commit 8ceddd60)
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

            # ---- Proj Layer (same for both versions) ----
            parameters["proj"] = {
                "weight": preprocess_linear_weight(torch_model.proj.weight, dtype=weight_dtype),
                "bias": preprocess_linear_bias(torch_model.proj.bias, dtype=weight_dtype),
            }

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "n_embed, n_head, attn_pdrop, resid_pdrop, input_shape",
    # ((128, 4, 0.1, 0.1, (1, 174, 128)),),  # case where weight padding is not required , higher pcc
    ((72, 4, 0.1, 0.1, (1, 174, 72)),),  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_optimized", [False, True])  # False uses old implementation, True uses optimized
def test_self_attn(
    device, n_embed, n_head, attn_pdrop, resid_pdrop, input_shape, input_dtype, weight_dtype, use_optimized
):
    x = torch.randn(input_shape)

    ref_layer = SelfAttention(
        n_embd=n_embed,
        n_head=n_head,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
    ).eval()
    ref_output = ref_layer(x)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_self_attn_preprocessor(device, weight_dtype, use_optimized=use_optimized),
        device=device,
    )
    tt_layer = TTSelfAttention(
        device,
        parameters,
        n_embed,
        n_head,
        dtype=weight_dtype,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        use_optimized=use_optimized,
    )
    tt_input = ttnn.from_torch(
        x, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_output = tt_layer(tt_input)
    tt_torch_output = tt2torch_tensor(tt_output)
    does_pass, pcc_message = check_with_pcc(ref_output, tt_torch_output, 0.95)

    logger.info(f"PCC: {pcc_message}")

    if does_pass:
        logger.info("SelfAttention Passed!")
    else:
        logger.warning("SelfAttention Failed!")

    assert does_pass, f"PCC check failed: {pcc_message}"
