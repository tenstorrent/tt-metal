# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights: building DeepSeekV3Weights from a state dict.

- test_prepare_dense_layer: one dense layer with random weights.
- test_prepare_moe_layer: one MoE layer with random weights.
- test_prepare_real_weights: placeholder for future real checkpoint testing.
"""

import pytest
import torch

from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3MoELayerWeights,
    prepare_weights,
)


def _layer_state_dict(layer_idx: int, *, is_moe: bool, seed: int = 42) -> dict[str, torch.Tensor]:
    """Build a minimal state_dict for a single layer (HF key convention, random weights)."""
    g = torch.Generator().manual_seed(seed)
    # HF linear weights are (out_features, in_features)
    state = {
        f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": torch.randn(1536, 7168, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": torch.randn(12288, 1536, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(576, 7168, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": torch.randn(16384, 512, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(7168, 8192, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(7168, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": torch.randn(1536, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": torch.randn(512, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": torch.randn(7168, generator=g, dtype=torch.bfloat16),
    }
    if is_moe:
        state[f"model.layers.{layer_idx}.mlp.gate.weight"] = torch.randn(256, 7168, generator=g, dtype=torch.bfloat16)
        state[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(256, 7168, generator=g, dtype=torch.bfloat16)
        state[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(256, 7168, generator=g, dtype=torch.bfloat16)
    return state


def test_prepare_dense_layer(device):
    """Build a single dense layer with random weights; verify type and shapes."""
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    state = _layer_state_dict(0, is_moe=False)
    weights = prepare_weights(
        state,
        device,
        num_layers=1,
        first_k_dense_replace=1,
    )
    assert len(weights.layers) == 1
    layer = weights.layers[0]
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)

    assert layer.q_a_proj.tensor_shape == (3584, 3072)  # packed
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.q_norm.tensor_shape == (1, 1536)
    assert layer.kv_norm.tensor_shape == (1, 512)
    assert layer.ffn_norm.tensor_shape == (1, 7168)
    assert layer.kv_b1_proj.tensor_shape == (8192, 512)
    assert layer.kv_b2_proj.tensor_shape == (512, 8192)


def test_prepare_moe_layer(device):
    """Build one dense and one MoE layer with random weights; verify MoE type and shapes."""
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    state = _layer_state_dict(0, is_moe=False, seed=42) | _layer_state_dict(1, is_moe=True, seed=43)
    weights = prepare_weights(
        state,
        device,
        num_layers=2,
        first_k_dense_replace=1,
    )
    assert len(weights.layers) == 2
    assert isinstance(weights.layers[0], DeepSeekV3DenseLayerWeights)
    layer = weights.layers[1]
    assert isinstance(layer, DeepSeekV3MoELayerWeights)

    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.gate_mm.tensor_shape == (7168, 256)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.q_norm.tensor_shape == (1, 1536)
    assert layer.kv_norm.tensor_shape == (1, 512)
    assert layer.ffn_norm.tensor_shape == (1, 7168)
    assert layer.kv_b1_proj.tensor_shape == (8192, 512)
    assert layer.kv_b2_proj.tensor_shape == (512, 8192)
    assert layer.shared_gate_proj.tensor_shape == (57344, 32)  # stacked
    assert layer.shared_up_proj.tensor_shape == (57344, 32)


@pytest.mark.skip(reason="Future: run with real HF checkpoint; not implemented yet.")
def test_prepare_real_weights(device):
    """Placeholder for testing prepare_weights with real model weights."""
    pytest.fail("Real-weight test not implemented yet.")
