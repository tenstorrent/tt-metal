# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights: building DeepSeekV3Weights from a state dict.

- test_prepare_dense_layer_single_layer: one dense layer with random weights.
- test_prepare_moe_layer_single_layer: one MoE layer with random weights.
- test_save_load_dense_layer_single_layer / test_save_load_moe_layer_single_layer: save then load one layer.
- test_prepare_dense_layer_single_layer_4x2: one dense layer on 4x2 mesh.
- test_prepare_moe_layer_single_layer_4x2: one MoE layer on 4x2 mesh.
- test_save_load_dense_layer_single_layer_4x2: save then load one dense layer on 4x2 submesh.
- test_save_load_moe_layer_single_layer_4x2: save then load one MoE layer on 4x2 submesh.
- test_load_4_layers_across_4_submeshes_4x2: prepare each of 4 layers on a different 4x2 submesh, save, then load each on same submesh (32 devices).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3MoELayerWeights,
    deallocate_weights,
    load_layer,
    prepare_weights,
    save_layer,
)


def _core_range_set_to_tuples(crs):
    """Normalize CoreRangeSet to comparable list of tuples for assertion."""
    return sorted(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in crs.ranges())


def _assert_overlapped_tensors_match(a: OverlappedTensor, b: OverlappedTensor) -> None:
    """Assert two OverlappedTensors have matching metadata (not fused_tensor identity)."""
    assert a.tensor_shape == b.tensor_shape
    assert a.shard_shape == b.shard_shape
    assert a.dtype == b.dtype
    assert a.tile_shape == b.tile_shape
    assert a.byte_offset == b.byte_offset
    assert _core_range_set_to_tuples(a.core_range_set) == _core_range_set_to_tuples(b.core_range_set)


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
        # shared down: HF (7168, 2048) per-device or (7168, 2048*tp) full logical
        shared_down_rows = 7168
        shared_down_cols = shared_hf[0]  # 256 per-device or 2048 full; for tp use 2048 * tp
        state[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(
            shared_down_rows, shared_down_cols, generator=g, dtype=torch.bfloat16
        )
        for e in range(32):
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                2048, 7168, generator=g, dtype=torch.bfloat16
            )
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                2048, 7168, generator=g, dtype=torch.bfloat16
            )
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                7168, 2048, generator=g, dtype=torch.bfloat16
            )
    else:
        # Dense MLP: HF (out, in) gate/up (18432, 7168), down (7168, 18432)
        state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.randn(
            18432, 7168, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.randn(
            18432, 7168, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.randn(
            7168, 18432, generator=g, dtype=torch.bfloat16
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
    assert layer.shared_gate_proj.tensor_shape is not None
    assert layer.shared_up_proj.tensor_shape is not None
    assert hasattr(layer, "routed_gate_proj") and layer.routed_gate_proj is not None
    assert hasattr(layer, "routed_up_proj") and layer.routed_up_proj is not None
    assert hasattr(layer, "routed_down_proj") and layer.routed_down_proj is not None


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
    assert hasattr(layer, "shared_down_proj")
    assert len(layer.routed_gate_proj) == 32
    assert len(layer.routed_up_proj) == 32
    assert len(layer.routed_down_proj) == 32


def test_save_load_dense_layer_single_layer(device, tmp_path):
    """Save one dense layer to disk, load it back, assert metadata and fused-tensor sharing."""
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")

    state = _layer_state_dict(0, is_moe=False)
    weights = prepare_weights(state, device, num_layers=1, first_k_dense_replace=1)
    save_layer(
        weights.layers[0],
        tmp_path,
        0,
        hf_model_name="test-dense-model",
        hf_state_dict_name="test-dense-state-dict.safetensors",
    )

    assert (tmp_path / "layer_000" / "manifest.json").exists()
    layer_dir = tmp_path / "layer_000"
    assert (layer_dir / "q_ab_kv_a.tensorbin").exists()
    assert (layer_dir / "o_proj_gate_mm_norms.tensorbin").exists()
    assert (layer_dir / "kv_b12.tensorbin").exists()
    assert (layer_dir / "gate_up.tensorbin").exists()
    assert (layer_dir / "routed_gate_proj.tensorbin").exists()
    assert (layer_dir / "routed_up_proj.tensorbin").exists()
    assert (layer_dir / "routed_down_proj.tensorbin").exists()

    deallocate_weights(weights)
    t0 = time.perf_counter()
    layer = load_layer(tmp_path, device, 0)
    elapsed = time.perf_counter() - t0
    logger.info("load_layer (dense, single device): {:.3f} s", elapsed)
    orig = weights.layers[0]
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)

    _assert_overlapped_tensors_match(orig.shared_gate_proj, layer.shared_gate_proj)
    _assert_overlapped_tensors_match(orig.shared_up_proj, layer.shared_up_proj)
    assert layer.routed_gate_proj.shape == orig.routed_gate_proj.shape
    assert layer.routed_up_proj.shape == orig.routed_up_proj.shape
    assert layer.routed_down_proj.shape == orig.routed_down_proj.shape

    _assert_overlapped_tensors_match(orig.q_a_proj, layer.q_a_proj)
    _assert_overlapped_tensors_match(orig.q_b_proj, layer.q_b_proj)
    _assert_overlapped_tensors_match(orig.kv_a_proj, layer.kv_a_proj)
    _assert_overlapped_tensors_match(orig.o_proj, layer.o_proj)
    _assert_overlapped_tensors_match(orig.attn_norm, layer.attn_norm)
    _assert_overlapped_tensors_match(orig.q_norm, layer.q_norm)
    _assert_overlapped_tensors_match(orig.kv_norm, layer.kv_norm)
    _assert_overlapped_tensors_match(orig.ffn_norm, layer.ffn_norm)
    _assert_overlapped_tensors_match(orig.kv_b1_proj, layer.kv_b1_proj)
    _assert_overlapped_tensors_match(orig.kv_b2_proj, layer.kv_b2_proj)

    # Same fusion group must share fused_tensor
    assert id(layer.q_a_proj.fused_tensor) == id(layer.q_b_proj.fused_tensor)
    assert id(layer.q_b_proj.fused_tensor) == id(layer.kv_a_proj.fused_tensor)
    assert id(layer.o_proj.fused_tensor) == id(layer.attn_norm.fused_tensor)
    assert id(layer.kv_b1_proj.fused_tensor) == id(layer.kv_b2_proj.fused_tensor)
    assert id(layer.shared_gate_proj.fused_tensor) == id(layer.shared_up_proj.fused_tensor)


def test_save_load_moe_layer_single_layer(device, tmp_path):
    """Save one MoE layer to disk, load it back, assert metadata and fused-tensor sharing."""
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    num_cores = device_grid.x * device_grid.y
    if num_cores < 128:
        pytest.skip(f"Device has {num_cores} compute cores; GATE_UP overlap spec requires 128 cores")

    state = _layer_state_dict(0, is_moe=True, seed=43)
    weights = prepare_weights(state, device, num_layers=1, first_k_dense_replace=0)
    save_layer(
        weights.layers[0],
        tmp_path,
        0,
        hf_model_name="test-moe-model",
        hf_state_dict_name="test-moe-state-dict.safetensors",
    )

    assert (tmp_path / "layer_000" / "manifest.json").exists()
    layer_dir = tmp_path / "layer_000"
    assert (layer_dir / "gate_up.tensorbin").exists()
    assert (layer_dir / "shared_down_proj.tensorbin").exists()
    experts_dir = layer_dir / "experts"
    for e in range(32):
        expert_dir = experts_dir / f"e_{e:03d}"
        assert (expert_dir / "gate_proj.tensorbin").exists()
        assert (expert_dir / "up_proj.tensorbin").exists()
        assert (expert_dir / "down_proj.tensorbin").exists()

    deallocate_weights(weights)
    t0 = time.perf_counter()
    layer = load_layer(tmp_path, device, 0)
    elapsed = time.perf_counter() - t0
    logger.info("load_layer (moe, single device): {:.3f} s", elapsed)
    orig = weights.layers[0]
    assert isinstance(layer, DeepSeekV3MoELayerWeights)

    _assert_overlapped_tensors_match(orig.q_a_proj, layer.q_a_proj)
    _assert_overlapped_tensors_match(orig.q_b_proj, layer.q_b_proj)
    _assert_overlapped_tensors_match(orig.kv_a_proj, layer.kv_a_proj)
    _assert_overlapped_tensors_match(orig.o_proj, layer.o_proj)
    _assert_overlapped_tensors_match(orig.gate_mm, layer.gate_mm)
    _assert_overlapped_tensors_match(orig.attn_norm, layer.attn_norm)
    _assert_overlapped_tensors_match(orig.q_norm, layer.q_norm)
    _assert_overlapped_tensors_match(orig.kv_norm, layer.kv_norm)
    _assert_overlapped_tensors_match(orig.ffn_norm, layer.ffn_norm)
    _assert_overlapped_tensors_match(orig.kv_b1_proj, layer.kv_b1_proj)
    _assert_overlapped_tensors_match(orig.kv_b2_proj, layer.kv_b2_proj)
    _assert_overlapped_tensors_match(orig.shared_gate_proj, layer.shared_gate_proj)
    _assert_overlapped_tensors_match(orig.shared_up_proj, layer.shared_up_proj)
    assert layer.shared_down_proj.shape == orig.shared_down_proj.shape
    assert len(layer.routed_gate_proj) == 32
    assert len(layer.routed_up_proj) == 32
    assert len(layer.routed_down_proj) == 32
    for e in range(32):
        assert layer.routed_gate_proj[e].shape == orig.routed_gate_proj[e].shape
        assert layer.routed_up_proj[e].shape == orig.routed_up_proj[e].shape
        assert layer.routed_down_proj[e].shape == orig.routed_down_proj[e].shape

    assert id(layer.shared_gate_proj.fused_tensor) == id(layer.shared_up_proj.fused_tensor)


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
    assert hasattr(layer, "shared_gate_proj") and layer.shared_gate_proj is not None
    assert hasattr(layer, "shared_up_proj") and layer.shared_up_proj is not None
    assert hasattr(layer, "routed_gate_proj") and layer.routed_gate_proj is not None
    assert hasattr(layer, "routed_up_proj") and layer.routed_up_proj is not None
    assert hasattr(layer, "routed_down_proj") and layer.routed_down_proj is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_save_load_dense_layer_single_layer_4x2(bh_2d_mesh_device, tmp_path):
    """Save one dense layer (4x2 submesh) to disk, load it back, assert metadata and fused-tensor sharing."""
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
    weights = prepare_weights(state, submesh, num_layers=1, first_k_dense_replace=1)
    orig = weights.layers[0]
    save_layer(
        orig,
        tmp_path,
        0,
        hf_model_name="test-dense-model-4x2",
        hf_state_dict_name="test-dense-state-dict.safetensors",
        device_mesh_shape=(4, 2),
    )

    assert (tmp_path / "layer_000" / "manifest.json").exists()
    layer_dir = tmp_path / "layer_000"
    assert (layer_dir / "q_ab_kv_a.tensorbin").exists()
    assert (layer_dir / "o_proj_gate_mm_norms.tensorbin").exists()
    assert (layer_dir / "kv_b12.tensorbin").exists()
    assert (layer_dir / "gate_up.tensorbin").exists()
    assert (layer_dir / "routed_gate_proj.tensorbin").exists()
    assert (layer_dir / "routed_up_proj.tensorbin").exists()
    assert (layer_dir / "routed_down_proj.tensorbin").exists()

    deallocate_weights(weights)
    t0 = time.perf_counter()
    layer = load_layer(tmp_path, submesh, 0)
    elapsed = time.perf_counter() - t0
    logger.info("load_layer (dense, 4x2 submesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)

    _assert_overlapped_tensors_match(orig.q_a_proj, layer.q_a_proj)
    _assert_overlapped_tensors_match(orig.q_b_proj, layer.q_b_proj)
    _assert_overlapped_tensors_match(orig.kv_a_proj, layer.kv_a_proj)
    _assert_overlapped_tensors_match(orig.o_proj, layer.o_proj)
    _assert_overlapped_tensors_match(orig.attn_norm, layer.attn_norm)
    _assert_overlapped_tensors_match(orig.q_norm, layer.q_norm)
    _assert_overlapped_tensors_match(orig.kv_norm, layer.kv_norm)
    _assert_overlapped_tensors_match(orig.ffn_norm, layer.ffn_norm)
    _assert_overlapped_tensors_match(orig.kv_b1_proj, layer.kv_b1_proj)
    _assert_overlapped_tensors_match(orig.kv_b2_proj, layer.kv_b2_proj)
    _assert_overlapped_tensors_match(orig.shared_gate_proj, layer.shared_gate_proj)
    _assert_overlapped_tensors_match(orig.shared_up_proj, layer.shared_up_proj)
    assert layer.routed_gate_proj.shape == orig.routed_gate_proj.shape
    assert layer.routed_up_proj.shape == orig.routed_up_proj.shape
    assert layer.routed_down_proj.shape == orig.routed_down_proj.shape

    assert id(layer.q_a_proj.fused_tensor) == id(layer.q_b_proj.fused_tensor)
    assert id(layer.q_b_proj.fused_tensor) == id(layer.kv_a_proj.fused_tensor)
    assert id(layer.o_proj.fused_tensor) == id(layer.attn_norm.fused_tensor)
    assert id(layer.kv_b1_proj.fused_tensor) == id(layer.kv_b2_proj.fused_tensor)
    assert id(layer.shared_gate_proj.fused_tensor) == id(layer.shared_up_proj.fused_tensor)


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
    assert hasattr(layer, "shared_down_proj")
    assert len(layer.routed_gate_proj) == 32
    assert len(layer.routed_up_proj) == 32
    assert len(layer.routed_down_proj) == 32


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_save_load_moe_layer_single_layer_4x2(bh_2d_mesh_device, tmp_path):
    """Save one MoE layer (4x2 submesh) to disk, load it back, assert metadata and fused-tensor sharing."""
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
        pytest.skip(f"Per-device has {num_cores} compute cores; GATE_UP overlap spec requires 128 cores")

    state = _layer_state_dict(0, is_moe=True, for_multi_device=True, seed=43)
    weights = prepare_weights(state, submesh, num_layers=1, first_k_dense_replace=0)
    orig = weights.layers[0]
    save_layer(
        orig,
        tmp_path,
        0,
        hf_model_name="test-moe-model-4x2",
        hf_state_dict_name="test-moe-state-dict.safetensors",
        device_mesh_shape=(4, 2),
    )

    assert (tmp_path / "layer_000" / "manifest.json").exists()
    layer_dir = tmp_path / "layer_000"
    assert (layer_dir / "gate_up.tensorbin").exists()
    assert (layer_dir / "shared_down_proj.tensorbin").exists()
    experts_dir = layer_dir / "experts"
    for e in range(32):
        expert_dir = experts_dir / f"e_{e:03d}"
        assert (expert_dir / "gate_proj.tensorbin").exists()
        assert (expert_dir / "up_proj.tensorbin").exists()
        assert (expert_dir / "down_proj.tensorbin").exists()

    deallocate_weights(weights)
    t0 = time.perf_counter()
    layer = load_layer(tmp_path, submesh, 0)
    elapsed = time.perf_counter() - t0
    logger.info("load_layer (moe, 4x2 submesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)

    _assert_overlapped_tensors_match(orig.q_a_proj, layer.q_a_proj)
    _assert_overlapped_tensors_match(orig.q_b_proj, layer.q_b_proj)
    _assert_overlapped_tensors_match(orig.kv_a_proj, layer.kv_a_proj)
    _assert_overlapped_tensors_match(orig.o_proj, layer.o_proj)
    _assert_overlapped_tensors_match(orig.gate_mm, layer.gate_mm)
    _assert_overlapped_tensors_match(orig.attn_norm, layer.attn_norm)
    _assert_overlapped_tensors_match(orig.q_norm, layer.q_norm)
    _assert_overlapped_tensors_match(orig.kv_norm, layer.kv_norm)
    _assert_overlapped_tensors_match(orig.ffn_norm, layer.ffn_norm)
    _assert_overlapped_tensors_match(orig.kv_b1_proj, layer.kv_b1_proj)
    _assert_overlapped_tensors_match(orig.kv_b2_proj, layer.kv_b2_proj)
    _assert_overlapped_tensors_match(orig.shared_gate_proj, layer.shared_gate_proj)
    _assert_overlapped_tensors_match(orig.shared_up_proj, layer.shared_up_proj)
    assert layer.shared_down_proj.shape == orig.shared_down_proj.shape
    assert len(layer.routed_gate_proj) == 32
    assert len(layer.routed_up_proj) == 32
    assert len(layer.routed_down_proj) == 32
    for e in range(32):
        assert layer.routed_gate_proj[e].shape == orig.routed_gate_proj[e].shape
        assert layer.routed_up_proj[e].shape == orig.routed_up_proj[e].shape
        assert layer.routed_down_proj[e].shape == orig.routed_down_proj[e].shape

    assert id(layer.shared_gate_proj.fused_tensor) == id(layer.shared_up_proj.fused_tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_load_4_layers_across_4_submeshes_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare each of 4 layers on a different 4x2 submesh, save them, then load each on the same submesh.

    Uses create_submeshes to get 4 disjoint (4x2) submeshes; requires 32 devices.
    Each layer is prepared on its own submesh to avoid OOM. Layers 0,1,2 are dense, layer 3 is MoE.
    """

    tmp_path = "/tmp/test-cache/"
    if not is_slow_dispatch():
        pytest.skip("prepare_weights tests require slow dispatch mode (TT_METAL_SLOW_DISPATCH_MODE=1)")
    num_submeshes = 4
    devices_per_submesh = 4 * 2
    num_devices_required = num_submeshes * devices_per_submesh  # 32
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices_required:
        pytest.skip(
            f"Test requires {num_devices_required} devices (4 submeshes x 4x2), "
            f"mesh has {bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]}"
        )
    submeshes = bh_2d_mesh_device.create_submeshes(ttnn.MeshShape((4, 2)))
    assert len(submeshes) >= num_submeshes, f"Expected at least {num_submeshes} submeshes"
    submesh0 = submeshes[0]
    device_grid = submesh0.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 10:
        pytest.skip(f"Per-device grid {device_grid.x}x{device_grid.y} too small for blitz decode (need 12x10+)")
    num_cores = device_grid.x * device_grid.y
    if num_cores < 128:
        pytest.skip(f"Per-device has {num_cores} compute cores; GATE_UP overlap spec requires 128 cores")

    num_layers = 4
    first_k_dense_replace = 3  # layers 0,1,2 dense; layer 3 MoE
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace
        submesh = submeshes[layer_idx]
        state = _layer_state_dict(
            layer_idx,
            is_moe=is_moe,
            for_multi_device=True,
            seed=42 + layer_idx,
        )
        # prepare_weights always looks up model.layers.0.* when num_layers=1; remap keys
        state_for_prepare = {k.replace(f"model.layers.{layer_idx}.", "model.layers.0."): v for k, v in state.items()}
        # first_k_dense_replace so the single layer (index 0) is dense or MoE
        first_k = 1 if layer_idx < first_k_dense_replace else 0
        t0 = time.perf_counter()
        weights = prepare_weights(
            state_for_prepare,
            submesh,
            num_layers=1,
            first_k_dense_replace=first_k,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "prepare_weights (layer %d, %s, on submesh %d): {:.3f} s",
            layer_idx,
            "moe" if is_moe else "dense",
            layer_idx,
            elapsed,
        )
        assert len(weights.layers) == 1
        save_layer(
            weights.layers[0],
            tmp_path,
            layer_idx,
            hf_model_name="test-4layer-model",
            hf_state_dict_name="test-4layer-state-dict.safetensors",
            device_mesh_shape=(4, 2),
        )
        deallocate_weights(weights)

    loaded = []
    t0 = time.time()
    for layer_idx in range(num_layers):
        submesh = submeshes[layer_idx]
        layer = load_layer(tmp_path, submesh, layer_idx)
        loaded.append(layer)
    elapsed = time.time() - t0
    logger.info(f"load_layer (4 layers, 4x2 submesh): {elapsed:.3f} s")

    assert isinstance(loaded[0], DeepSeekV3DenseLayerWeights)
    assert isinstance(loaded[1], DeepSeekV3DenseLayerWeights)
    assert isinstance(loaded[2], DeepSeekV3DenseLayerWeights)
    assert isinstance(loaded[3], DeepSeekV3MoELayerWeights)
    for i in range(3):
        assert hasattr(loaded[i], "shared_gate_proj") and loaded[i].shared_gate_proj is not None
        assert hasattr(loaded[i], "routed_gate_proj") and loaded[i].routed_gate_proj is not None
        assert loaded[i].routed_gate_proj.shape is not None
    assert len(loaded[3].routed_gate_proj) == 32
    assert len(loaded[3].routed_up_proj) == 32
    assert len(loaded[3].routed_down_proj) == 32
    assert loaded[3].shared_down_proj is not None
