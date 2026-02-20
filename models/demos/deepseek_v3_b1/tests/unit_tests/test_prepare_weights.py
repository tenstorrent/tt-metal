# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights: building DeepSeekV3Weights from a state dict.

- test_prepare_dense_layer_single_layer: one dense layer with random weights.
- test_prepare_moe_layer_single_layer: one MoE layer with random weights.
- test_prepare_dense_layer_single_layer_4x2: one dense layer on 4x2 mesh.
- test_prepare_moe_layer_single_layer_4x2: one MoE layer on 4x2 mesh.
- test_prepare_real_weights: placeholder for future real checkpoint testing.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3MoELayerWeights,
    prepare_weights,
)

# HF state dict shapes (out_features, in_features) for linears; see DEEPSEEK_PREPARE_WEIGHTS_DESIGN_DOC.md §5.
# Per-device = one shard for single-device; blitz expects these when device is single.
# Full logical = full model; pass to prepare_weights when device is 4x2 mesh; blitz shards internally.
HF_Q_B_PER_DEVICE = (12288, 1536)
HF_Q_B_FULL_LOGICAL = (24576, 1536)
HF_O_PROJ_PER_DEVICE = (7168, 8192)
HF_O_PROJ_FULL_LOGICAL = (7168, 16384)
HF_KV_B_PER_DEVICE = (16384, 512)
HF_KV_B_FULL_LOGICAL = (32768, 512)
HF_SHARED_GATE_UP_PER_DEVICE = (256, 7168)
HF_SHARED_GATE_UP_FULL_LOGICAL = (2048, 7168)


def _layer_state_dict(
    layer_idx: int,
    *,
    is_moe: bool,
    for_multi_device: bool = False,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build a minimal state_dict for one layer (HF key convention, random weights).

    For single device: use for_multi_device=False so tensors are per-device shards
    (sliced from full logical); prepare_weights uses them as-is.
    For 4x2 mesh: use for_multi_device=True so tensors are full logical shapes;
    prepare_weights passes them to blitz, which shards across the mesh.
    """
    g = torch.Generator().manual_seed(seed)
    q_b_hf = HF_Q_B_FULL_LOGICAL if for_multi_device else HF_Q_B_PER_DEVICE
    o_proj_hf = HF_O_PROJ_FULL_LOGICAL if for_multi_device else HF_O_PROJ_PER_DEVICE
    kv_b_hf = HF_KV_B_FULL_LOGICAL if for_multi_device else HF_KV_B_PER_DEVICE
    shared_hf = HF_SHARED_GATE_UP_FULL_LOGICAL if for_multi_device else HF_SHARED_GATE_UP_PER_DEVICE

    state = {
        f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": torch.randn(
            1536, 7168, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": torch.randn(*q_b_hf, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(
            576, 7168, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": torch.randn(
            *kv_b_hf, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(*o_proj_hf, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(7168, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": torch.randn(
            1536, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": torch.randn(
            512, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": torch.randn(
            7168, generator=g, dtype=torch.bfloat16
        ),
    }
    if is_moe:
        state[f"model.layers.{layer_idx}.mlp.gate.weight"] = torch.randn(256, 7168, generator=g, dtype=torch.bfloat16)
        state[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
    return state


def test_prepare_dense_layer_single_layer(device):
    """Build a single dense layer with random weights; verify type and shapes."""
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    state = _layer_state_dict(0, is_moe=False)
    t0 = time.perf_counter()
    weights = prepare_weights(
        state,
        device,
        num_layers=1,
        first_k_dense_replace=1,
    )
    elapsed = time.perf_counter() - t0
    logger.info("prepare_weights (1 dense layer): {:.3f} s", elapsed)
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


def test_prepare_moe_layer_single_layer(device):
    """Build a single MoE layer with random weights; verify type and shapes.

    We prepare only one layer to avoid L1 OOM (two full blitz layers exceed
    device L1 capacity on current hardware).
    """
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    num_cores = device_grid.x * device_grid.y
    if num_cores < 128:
        pytest.skip(f"Device has {num_cores} compute cores; GATE_UP overlap spec requires 128 cores (13x10 grid)")
    state = _layer_state_dict(0, is_moe=True, seed=43)
    t0 = time.perf_counter()
    weights = prepare_weights(
        state,
        device,
        num_layers=1,
        first_k_dense_replace=0,
    )
    elapsed = time.perf_counter() - t0
    logger.info("prepare_weights (1 MoE layer): {:.3f} s", elapsed)
    assert len(weights.layers) == 1
    layer = weights.layers[0]
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
    # tensor_shape is logical (unsharded); gate/up are HEIGHT_SHARDED as (57344, 32) on device
    assert layer.shared_gate_proj.tensor_shape == (7168, 256)
    assert layer.shared_up_proj.tensor_shape == (7168, 256)


# ---------------------------------------------------------------------------
# Multi-device (4x2 grid) tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_dense_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one dense layer on 4x2 mesh; verify type and shapes (MLA TP=2)."""
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    num_devices = 4 * 2
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires 8 devices (4x2 mesh)")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    device_grid = submesh.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Per-device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    state = _layer_state_dict(0, is_moe=False, for_multi_device=True)
    t0 = time.perf_counter()
    weights = prepare_weights(
        state,
        submesh,
        num_layers=1,
        first_k_dense_replace=1,
    )
    elapsed = time.perf_counter() - t0
    logger.info("prepare_weights (1 dense layer, 4x2 mesh): {:.3f} s", elapsed)
    assert len(weights.layers) == 1
    layer = weights.layers[0]
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.q_norm.tensor_shape == (1, 1536)
    assert layer.kv_norm.tensor_shape == (1, 512)
    assert layer.ffn_norm.tensor_shape == (1, 7168)
    assert layer.kv_b1_proj.tensor_shape == (8192, 512)
    assert layer.kv_b2_proj.tensor_shape == (512, 8192)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one MoE layer on 4x2 mesh; verify type and shapes (MLA TP=2, MoE TP=8)."""
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    num_devices = 4 * 2
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires 8 devices (4x2 mesh)")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    device_grid = submesh.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Per-device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    num_cores = device_grid.x * device_grid.y
    if num_cores < 128:
        pytest.skip(f"Per-device has {num_cores} compute cores; GATE_UP overlap spec requires 128 cores (13x10 grid)")
    state = _layer_state_dict(0, is_moe=True, for_multi_device=True, seed=43)
    t0 = time.perf_counter()
    weights = prepare_weights(
        state,
        submesh,
        num_layers=1,
        first_k_dense_replace=0,
    )
    elapsed = time.perf_counter() - t0
    logger.info("prepare_weights (1 MoE layer, 4x2 mesh): {:.3f} s", elapsed)
    assert len(weights.layers) == 1
    layer = weights.layers[0]
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
    assert layer.shared_gate_proj.tensor_shape == (7168, 256)
    assert layer.shared_up_proj.tensor_shape == (7168, 256)


@pytest.mark.skip(reason="Future: run with real HF checkpoint; not implemented yet.")
def test_prepare_real_weights(device):
    """Placeholder for testing prepare_weights with real model weights."""
    pytest.fail("Real-weight test not implemented yet.")
