# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.deepseek_v3.utils.hf_model_utils import load_weight_from_weights_dict

REFERENCE_WEIGHT_KEYS = [
    # Embedding + LM head (full model endpoints)
    "model.embed_tokens.weight",
    "lm_head.weight",
    # Dense decoder block weights
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    # MoE decoder block weights
    "model.layers.3.mlp.experts.0.gate_proj.weight",
]


@pytest.mark.parametrize("weight_name", REFERENCE_WEIGHT_KEYS)
def test_loaded_dequantized_weight_matches_reference_tensor(state_dict, weight_name):
    if weight_name not in state_dict:
        pytest.skip(f"Checkpoint does not contain '{weight_name}'")

    # The dequantized checkpoint should not have fp8 scales for these weights.
    assert f"{weight_name}_scale_inv" not in state_dict

    reference_weight = state_dict[weight_name]
    assert reference_weight.dtype != torch.float8_e4m3fn

    load_weight = load_weight_from_weights_dict(state_dict)
    target_tensor = torch.empty_like(reference_weight)
    loaded_tensor = load_weight(weight_name, target_tensor)

    assert loaded_tensor is target_tensor
    torch.testing.assert_close(loaded_tensor, reference_weight, rtol=0.0, atol=0.0)


def test_loaded_dequantized_weight_matches_reference_with_target_dtype_cast(state_dict):
    weight_name = "model.layers.0.mlp.down_proj.weight"
    if weight_name not in state_dict:
        pytest.skip(f"Checkpoint does not contain '{weight_name}'")

    reference_weight = state_dict[weight_name]
    assert reference_weight.dtype != torch.float8_e4m3fn

    load_weight = load_weight_from_weights_dict(state_dict)
    target_tensor = torch.empty(reference_weight.shape, dtype=torch.float32)
    loaded_tensor = load_weight(weight_name, target_tensor)

    assert loaded_tensor is target_tensor
    assert loaded_tensor.dtype == torch.float32
    torch.testing.assert_close(loaded_tensor, reference_weight.to(torch.float32), rtol=0.0, atol=0.0)
