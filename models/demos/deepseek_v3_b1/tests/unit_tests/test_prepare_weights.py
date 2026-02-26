# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights: per-layer prepare/save/load on 4x2 mesh.

- test_prepare_dense_layer_single_layer_4x2 / test_prepare_moe_layer_single_layer_4x2: one layer on 4x2 mesh.
- test_save_load_dense_layer_single_layer_4x2 / test_save_load_moe_layer_single_layer_4x2: save then load one layer on 4x2 submesh.
- test_load_4_layers_across_4_submeshes_4x2: prepare each of 4 layers on a different 4x2 submesh, save, then load each on same submesh (32 devices).
- test_prepare_attention_weights_*_4x2, test_prepare_shared_expert_weights_*_4x2, test_prepare_routed_expert_weights_*_4x2: per-group prepare on 4x2 mesh.
- test_incremental_save_load_dense_4x2 / test_incremental_save_load_moe_4x2: incremental save then load on 4x2 mesh.
- test_dump_load_routed_expert_weights_4x2: dump/load routed expert weights on 4x2 mesh.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights, OverlappedTensor
from models.demos.deepseek_v3_b1.prepare_weights import (
    AttentionWeights,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    SharedExpertWeights,
    load_dense_decoder_layer,
    load_embedding_weights,
    load_lm_head_weights,
    load_moe_decoder_layer,
    prepare_attention_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
    save_attention_weights,
    save_decoder_layer,
    save_embedding_weights,
    save_lm_head_weights,
    save_routed_expert_weights,
    save_shared_expert_weights,
)


def _deallocate_layer(layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights) -> None:
    """Deallocate all tensors in a single decoder layer (for tests that save then load)."""
    seen: set[int] = set()
    for f in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "o_proj",
        "attn_norm",
        "q_norm",
        "kv_norm",
        "ffn_norm",
        "kv_b1_proj",
        "kv_b2_proj",
        "shared_gate_proj",
        "shared_up_proj",
    ):
        ot = getattr(layer, f, None)
        if ot is not None and hasattr(ot, "fused_tensor"):
            fid = id(ot.fused_tensor)
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)
    ttnn.deallocate(layer.shared_down_proj, force=True)
    if isinstance(layer, DeepSeekV3MoELayerWeights):
        ttnn.deallocate(layer.gate_bias, force=True)
        for t in layer.routed_gate_proj:
            ttnn.deallocate(t, force=True)
        for t in layer.routed_up_proj:
            ttnn.deallocate(t, force=True)
        for t in layer.routed_down_proj:
            ttnn.deallocate(t, force=True)
    else:
        ttnn.deallocate(layer.routed_gate_proj, force=True)
        ttnn.deallocate(layer.routed_up_proj, force=True)
        ttnn.deallocate(layer.routed_down_proj, force=True)


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


def _assert_on_device(tensor: ttnn.Tensor) -> None:
    """Assert the tensor storage type is DEVICE."""
    assert tensor.storage_type() == ttnn.StorageType.DEVICE, f"Expected DEVICE storage, got {tensor.storage_type()}"


def _assert_topology(tensor: ttnn.Tensor, expected_placements: list) -> None:
    """Assert the tensor topology placements match expected."""
    actual = list(tensor.tensor_topology().placements())
    assert len(actual) == len(expected_placements), f"Expected {len(expected_placements)} placements, got {len(actual)}"
    for a, e in zip(actual, expected_placements):
        assert type(a) == type(e), f"Placement type mismatch: {a} vs {e}"
        if isinstance(e, ttnn.PlacementShard):
            assert a.dim == e.dim, f"Shard dim mismatch: {a.dim} vs {e.dim}"


def _skip_unless_4x2_mesh(bh_2d_mesh_device):
    """Skip test if mesh device does not have enough devices for a 4x2 submesh."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires 8 devices (4x2 mesh)")


# Expected placements for 4x2 mesh (mla_tp=2, moe_tp=8)
_PLACEMENTS_SHARD_NONE_1 = [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]
_PLACEMENTS_SHARD_NONE_0 = [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)]
_PLACEMENTS_SHARD_0_1 = [ttnn.PlacementShard(0), ttnn.PlacementShard(1)]
_PLACEMENTS_REPLICATE = [ttnn.PlacementReplicate()]


def _assert_layer_on_device_with_topology(
    layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights,
) -> None:
    """Assert all tensors in a loaded layer are on device and have correct topology for 4x2 mesh."""
    seen_fused: set[int] = set()
    # Check fusion groups via one representative OverlappedTensor per group
    # q_ab_kv_a
    _assert_on_device(layer.q_a_proj.fused_tensor)
    fid = id(layer.q_a_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.q_a_proj.fused_tensor, _PLACEMENTS_SHARD_NONE_1)
    # o_proj_gate_mm_norms
    _assert_on_device(layer.o_proj.fused_tensor)
    fid = id(layer.o_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.o_proj.fused_tensor, _PLACEMENTS_SHARD_NONE_1)
    # kv_b12
    _assert_on_device(layer.kv_b1_proj.fused_tensor)
    fid = id(layer.kv_b1_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.kv_b1_proj.fused_tensor, _PLACEMENTS_SHARD_NONE_0)
    # gate_up
    _assert_on_device(layer.shared_gate_proj.fused_tensor)
    fid = id(layer.shared_gate_proj.fused_tensor)
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.shared_gate_proj.fused_tensor, _PLACEMENTS_SHARD_0_1)
    # Standalone: shared_down_proj
    _assert_on_device(layer.shared_down_proj)
    _assert_topology(layer.shared_down_proj, _PLACEMENTS_SHARD_0_1)
    # Routed experts
    if isinstance(layer, DeepSeekV3DenseLayerWeights):
        _assert_on_device(layer.routed_gate_proj)
        _assert_on_device(layer.routed_up_proj)
        _assert_on_device(layer.routed_down_proj)
        _assert_topology(layer.routed_gate_proj, _PLACEMENTS_SHARD_0_1)
        _assert_topology(layer.routed_up_proj, _PLACEMENTS_SHARD_0_1)
        _assert_topology(layer.routed_down_proj, _PLACEMENTS_SHARD_0_1)
    else:
        assert isinstance(layer, DeepSeekV3MoELayerWeights)
        _assert_topology(layer.gate_bias, _PLACEMENTS_REPLICATE)
        for e in range(len(layer.routed_gate_proj)):
            _assert_on_device(layer.routed_gate_proj[e])
            _assert_on_device(layer.routed_up_proj[e])
            _assert_on_device(layer.routed_down_proj[e])
            _assert_topology(layer.routed_gate_proj[e], _PLACEMENTS_REPLICATE)
            _assert_topology(layer.routed_up_proj[e], _PLACEMENTS_REPLICATE)
            _assert_topology(layer.routed_down_proj[e], _PLACEMENTS_REPLICATE)


# HF state dict shapes (out_features, in_features) for linears; full logical for 4x2 mesh. See DEEPSEEK_PREPARE_WEIGHTS_DESIGN_DOC.md §5.
HF_Q_B_FULL_LOGICAL = (24576, 1536)
HF_O_PROJ_FULL_LOGICAL = (7168, 16384)
HF_KV_B_FULL_LOGICAL = (32768, 512)
HF_SHARED_GATE_UP_FULL_LOGICAL = (2048, 7168)

# Don't use all the experts in tests to avoid taking too long
NUM_ROUTED_EXPERTS = 4


def _layer_state_dict(
    layer_idx: int,
    *,
    is_moe: bool,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build a minimal state_dict for one layer (HF key convention, random weights).

    Uses full logical shapes for 4x2 mesh; prepare_weights passes them to blitz, which shards across the mesh.
    """
    g = torch.Generator().manual_seed(seed)
    q_b_hf = HF_Q_B_FULL_LOGICAL
    o_proj_hf = HF_O_PROJ_FULL_LOGICAL
    kv_b_hf = HF_KV_B_FULL_LOGICAL
    shared_hf = HF_SHARED_GATE_UP_FULL_LOGICAL

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
        state[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = torch.randn(
            256, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        # shared down: HF (7168, 2048*tp) full logical
        shared_down_rows = 7168
        shared_down_cols = shared_hf[0]  # 2048 for 4x2
        state[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(
            shared_down_rows, shared_down_cols, generator=g, dtype=torch.bfloat16
        )
        for e in range(NUM_ROUTED_EXPERTS):
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


def _add_global_weights(state: dict[str, torch.Tensor], seed: int = 42) -> None:
    """Add embedding, final norm, and lm_head to state (in place)."""
    g = torch.Generator().manual_seed(seed)
    state["model.embed_tokens.weight"] = torch.randn(129280, 7168, generator=g, dtype=torch.bfloat16)
    state["model.norm.weight"] = torch.randn(7168, generator=g, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(129280, 7168, generator=g, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_attention_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare attention weights only for a dense layer on 4x2 mesh; verify shapes and fusion group sharing."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    attn = prepare_attention_weights(bdw, state, 0, is_moe=False)
    assert attn.gate_mm is None
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    assert attn.q_b_proj.tensor_shape == (1536, 12288)
    assert attn.kv_a_proj.tensor_shape == (7168, 576)
    assert attn.o_proj.tensor_shape == (8192, 7168)
    assert attn.attn_norm.tensor_shape == (1, 7168)
    assert attn.kv_b1_proj.tensor_shape == (8192, 512)
    assert attn.kv_b2_proj.tensor_shape == (512, 8192)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_attention_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare attention weights only for an MoE layer on 4x2 mesh; verify shapes and gate_mm present."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    attn = prepare_attention_weights(bdw, state, 0, is_moe=True)
    assert attn.gate_mm is not None
    assert attn.gate_mm.tensor_shape == (7168, 256)
    assert attn.gate_bias is not None
    assert attn.gate_bias.shape == (16, 16)
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    assert attn.o_proj.tensor_shape == (8192, 7168)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_shared_expert_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare shared expert weights only for a dense layer on 4x2 mesh; verify shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    shared = prepare_shared_expert_weights(bdw, state, 0, is_moe=False)
    assert shared.shared_gate_proj.tensor_shape is not None
    assert shared.shared_up_proj.tensor_shape is not None
    assert shared.shared_down_proj.shape is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_shared_expert_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare shared expert weights only for an MoE layer on 4x2 mesh; verify shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    shared = prepare_shared_expert_weights(bdw, state, 0, is_moe=True)
    assert shared.shared_gate_proj.tensor_shape == (7168, 256)
    assert shared.shared_up_proj.tensor_shape == (7168, 256)
    assert shared.shared_down_proj.shape is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare routed expert weights only for a dense layer on 4x2 mesh; verify shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    routed = prepare_routed_expert_weights(bdw, state, 0, is_moe=False)
    assert isinstance(routed, DenseRoutedExpertWeights)
    assert routed.routed_gate_proj.shape is not None
    assert routed.routed_up_proj.shape is not None
    assert routed.routed_down_proj.shape is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare routed expert weights only for an MoE layer on 4x2 mesh; verify shapes and expert count."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    routed = prepare_routed_expert_weights(bdw, state, 0, is_moe=True, num_routed_experts=NUM_ROUTED_EXPERTS)
    assert isinstance(routed, MoERoutedExpertWeights)
    assert len(routed.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(routed.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(routed.routed_down_proj) == NUM_ROUTED_EXPERTS


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_incremental_save_load_dense_4x2(bh_2d_mesh_device, tmp_path):
    """Save dense layer on 4x2 via separate save_attention_weights, save_shared_expert_weights, save_routed_expert_weights; load_dense_decoder_layer and verify."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load_dense_decoder_layer requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    layer = prepare_dense_layer_weights(bdw, state, 0)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)
    attn = AttentionWeights(
        q_a_proj=layer.q_a_proj,
        q_b_proj=layer.q_b_proj,
        kv_a_proj=layer.kv_a_proj,
        o_proj=layer.o_proj,
        gate_mm=None,
        attn_norm=layer.attn_norm,
        q_norm=layer.q_norm,
        kv_norm=layer.kv_norm,
        ffn_norm=layer.ffn_norm,
        kv_b1_proj=layer.kv_b1_proj,
        kv_b2_proj=layer.kv_b2_proj,
        gate_bias=None,
    )
    shared = SharedExpertWeights(
        shared_gate_proj=layer.shared_gate_proj,
        shared_up_proj=layer.shared_up_proj,
        shared_down_proj=layer.shared_down_proj,
    )
    routed = DenseRoutedExpertWeights(
        routed_gate_proj=layer.routed_gate_proj,
        routed_up_proj=layer.routed_up_proj,
        routed_down_proj=layer.routed_down_proj,
    )
    manifest_kw = dict(
        hf_model_name="test-incremental-dense-4x2",
        hf_state_dict_name="test-incremental-dense.safetensors",
        device_mesh_shape=(4, 2),
    )
    save_attention_weights(attn, tmp_path, 0, is_moe=False, **manifest_kw)
    save_shared_expert_weights(shared, tmp_path, 0, is_moe=False, **manifest_kw)
    save_routed_expert_weights(routed, tmp_path, 0, is_moe=False, **manifest_kw)
    expected_routed_shape = layer.routed_gate_proj.shape
    _deallocate_layer(layer)
    loaded = load_dense_decoder_layer(tmp_path, submesh, 0)
    assert isinstance(loaded, DeepSeekV3DenseLayerWeights)
    _assert_overlapped_tensors_match(layer.q_a_proj, loaded.q_a_proj)
    _assert_overlapped_tensors_match(layer.shared_gate_proj, loaded.shared_gate_proj)
    assert loaded.routed_gate_proj.shape == expected_routed_shape
    _assert_layer_on_device_with_topology(loaded)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_incremental_save_load_moe_4x2(bh_2d_mesh_device, tmp_path):
    """Save MoE layer on 4x2 via separate save_attention_weights, save_shared_expert_weights, save_routed_expert_weights; load_moe_decoder_layer and verify."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load_moe_decoder_layer requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    layer = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    attn = AttentionWeights(
        q_a_proj=layer.q_a_proj,
        q_b_proj=layer.q_b_proj,
        kv_a_proj=layer.kv_a_proj,
        o_proj=layer.o_proj,
        gate_mm=layer.gate_mm,
        attn_norm=layer.attn_norm,
        q_norm=layer.q_norm,
        kv_norm=layer.kv_norm,
        ffn_norm=layer.ffn_norm,
        kv_b1_proj=layer.kv_b1_proj,
        kv_b2_proj=layer.kv_b2_proj,
        gate_bias=layer.gate_bias,
    )
    shared = SharedExpertWeights(
        shared_gate_proj=layer.shared_gate_proj,
        shared_up_proj=layer.shared_up_proj,
        shared_down_proj=layer.shared_down_proj,
    )
    routed = MoERoutedExpertWeights(
        routed_gate_proj=layer.routed_gate_proj,
        routed_up_proj=layer.routed_up_proj,
        routed_down_proj=layer.routed_down_proj,
    )
    manifest_kw = dict(
        hf_model_name="test-incremental-moe-4x2",
        hf_state_dict_name="test-incremental-moe.safetensors",
        device_mesh_shape=(4, 2),
    )
    save_attention_weights(attn, tmp_path, 0, is_moe=True, **manifest_kw)
    save_shared_expert_weights(shared, tmp_path, 0, is_moe=True, **manifest_kw)
    save_routed_expert_weights(routed, tmp_path, 0, is_moe=True, **manifest_kw)
    expected_routed_expert_shape = layer.routed_gate_proj[0].shape
    _deallocate_layer(layer)
    loaded = load_moe_decoder_layer(tmp_path, submesh, 0)
    assert isinstance(loaded, DeepSeekV3MoELayerWeights)
    _assert_overlapped_tensors_match(layer.gate_mm, loaded.gate_mm)
    assert loaded.gate_bias.shape == layer.gate_bias.shape
    _assert_overlapped_tensors_match(layer.shared_gate_proj, loaded.shared_gate_proj)
    assert len(loaded.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert loaded.routed_gate_proj[0].shape == expected_routed_expert_shape
    _assert_layer_on_device_with_topology(loaded)


def _moe_routed_expert_stacked_tensors(seed: int = 43):
    """Build (num_experts, K, N) stacked gate/up and (num_experts, K_down, N_down) down for get_tt_moe_routed_expert_weights."""
    g = torch.Generator().manual_seed(seed)
    # MoE expert shapes: gate/up (K=7168, N=2048), down (2048, 7168)
    gate_stacked = torch.randn(NUM_ROUTED_EXPERTS, 7168, 2048, generator=g, dtype=torch.bfloat16)
    up_stacked = torch.randn(NUM_ROUTED_EXPERTS, 7168, 2048, generator=g, dtype=torch.bfloat16)
    down_stacked = torch.randn(NUM_ROUTED_EXPERTS, 2048, 7168, generator=g, dtype=torch.bfloat16)
    return gate_stacked, up_stacked, down_stacked


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_dump_load_routed_expert_weights_4x2(bh_2d_mesh_device, tmp_path):
    """Test dump/load round-trip for MoE routed expert weights on 4x2 mesh. Uses BlitzDecodeWeights directly so it can run in slow dispatch."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    logger.info("Testing dump/load round-trip for MoE routed expert weights on 4x2 mesh...")
    gate_stacked, up_stacked, down_stacked = _moe_routed_expert_stacked_tensors(seed=43)
    bdw = BlitzDecodeWeights(submesh)
    routed_gate_proj, routed_up_proj, routed_down_proj = bdw.get_tt_moe_routed_expert_weights(
        gate_stacked, up_stacked, down_stacked
    )

    # Capture shapes before dump
    expected_gate_shapes = [routed_gate_proj[e].shape for e in range(NUM_ROUTED_EXPERTS)]
    expected_up_shapes = [routed_up_proj[e].shape for e in range(NUM_ROUTED_EXPERTS)]
    expected_down_shapes = [routed_down_proj[e].shape for e in range(NUM_ROUTED_EXPERTS)]

    # Dump only routed experts (same layout as save_layer for MoE)
    layer_dir = tmp_path / "layer_000"
    experts_dir = layer_dir / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)
    for e in range(NUM_ROUTED_EXPERTS):
        logger.info("Dumping expert {}...", e)
        expert_dir = experts_dir / f"e_{e:03d}"
        expert_dir.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(expert_dir / "gate_proj.tensorbin", routed_gate_proj[e])
        ttnn.dump_tensor(expert_dir / "up_proj.tensorbin", routed_up_proj[e])
        ttnn.dump_tensor(expert_dir / "down_proj.tensorbin", routed_down_proj[e])

    # Allow original tensors to be freed; load back onto the same submesh
    del routed_gate_proj, routed_up_proj, routed_down_proj
    routed_gate_proj = []
    routed_up_proj = []
    routed_down_proj = []
    logger.info("Loading routed experts back onto the same submesh...")
    t0 = time.perf_counter()
    for e in range(NUM_ROUTED_EXPERTS):
        expert_dir = experts_dir / f"e_{e:03d}"
        logger.info("Loading expert {}...", e)
        routed_gate_proj.append(ttnn.load_tensor(expert_dir / "gate_proj.tensorbin", device=submesh))
        routed_up_proj.append(ttnn.load_tensor(expert_dir / "up_proj.tensorbin", device=submesh))
        routed_down_proj.append(ttnn.load_tensor(expert_dir / "down_proj.tensorbin", device=submesh))
    elapsed = time.perf_counter() - t0
    logger.info("Loaded routed experts back onto the same submesh in {:.3f}s", elapsed)

    assert len(routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(routed_down_proj) == NUM_ROUTED_EXPERTS
    for e in range(NUM_ROUTED_EXPERTS):
        assert routed_gate_proj[e].shape == expected_gate_shapes[e]
        assert routed_up_proj[e].shape == expected_up_shapes[e]
        assert routed_down_proj[e].shape == expected_down_shapes[e]
        _assert_on_device(routed_gate_proj[e])
        _assert_on_device(routed_up_proj[e])
        _assert_on_device(routed_down_proj[e])
        _assert_topology(routed_gate_proj[e], _PLACEMENTS_REPLICATE)
        _assert_topology(routed_up_proj[e], _PLACEMENTS_REPLICATE)
        _assert_topology(routed_down_proj[e], _PLACEMENTS_REPLICATE)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_dense_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one dense layer on 4x2 mesh; verify type and shapes (MLA TP=2)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    t0 = time.perf_counter()
    layer = prepare_dense_layer_weights(bdw, state, 0)
    elapsed = time.perf_counter() - t0
    logger.info("prepare_dense_layer_weights (1 dense layer, 4x2 mesh): {:.3f} s", elapsed)
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
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load_dense_decoder_layer requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    t0 = time.perf_counter()
    orig = prepare_dense_layer_weights(bdw, state, 0)
    elapsed = time.perf_counter() - t0
    logger.info("prepare_dense_layer_weights (1 dense layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(orig, DeepSeekV3DenseLayerWeights)
    assert orig.q_a_proj.tensor_shape == (3584, 3072)
    assert orig.q_b_proj.tensor_shape == (1536, 12288)
    assert orig.kv_a_proj.tensor_shape == (7168, 576)
    assert orig.o_proj.tensor_shape == (8192, 7168)
    assert orig.attn_norm.tensor_shape == (1, 7168)
    assert orig.q_norm.tensor_shape == (1, 1536)
    assert orig.kv_norm.tensor_shape == (1, 512)
    assert orig.ffn_norm.tensor_shape == (1, 7168)
    assert orig.kv_b1_proj.tensor_shape == (8192, 512)
    assert orig.kv_b2_proj.tensor_shape == (512, 8192)
    assert hasattr(orig, "shared_gate_proj") and orig.shared_gate_proj is not None
    assert hasattr(orig, "shared_up_proj") and orig.shared_up_proj is not None
    assert hasattr(orig, "routed_gate_proj") and orig.routed_gate_proj is not None
    assert hasattr(orig, "routed_up_proj") and orig.routed_up_proj is not None
    assert hasattr(orig, "routed_down_proj") and orig.routed_down_proj is not None
    # Early access to q_ab_kv_a fused_tensor (same as first tensor touched in save_layer)
    logger.info("Early access: touching q_ab_kv_a fused_tensor (orig.q_a_proj.fused_tensor)...")
    q_ab_kv_a_fused = orig.q_a_proj.fused_tensor
    _ = q_ab_kv_a_fused.shape
    logger.info("Early access: got shape {}", q_ab_kv_a_fused.shape)
    save_decoder_layer(
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

    _deallocate_layer(orig)
    t0 = time.perf_counter()
    layer = load_dense_decoder_layer(tmp_path, submesh, 0)
    elapsed = time.perf_counter() - t0
    logger.info("load_dense_decoder_layer (dense, 4x2 submesh): {:.3f} s", elapsed)
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
    _assert_layer_on_device_with_topology(layer)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one MoE layer on 4x2 mesh; verify type and shapes (MLA TP=2, MoE TP=8)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    logger.info(f"State dict prepared")
    t0 = time.perf_counter()
    logger.info(f"Preparing weights...")
    layer = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
    logger.info(f"Weights prepared")
    elapsed = time.perf_counter() - t0
    logger.info("prepare_moe_layer_weights (1 MoE layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.gate_mm.tensor_shape == (7168, 256)
    assert layer.gate_bias.shape == (16, 16)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.q_norm.tensor_shape == (1, 1536)
    assert layer.kv_norm.tensor_shape == (1, 512)
    assert layer.ffn_norm.tensor_shape == (1, 7168)
    assert layer.kv_b1_proj.tensor_shape == (8192, 512)
    assert layer.kv_b2_proj.tensor_shape == (512, 8192)
    assert layer.shared_gate_proj.tensor_shape == (7168, 256)
    assert layer.shared_up_proj.tensor_shape == (7168, 256)
    assert hasattr(layer, "shared_down_proj")
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_down_proj) == NUM_ROUTED_EXPERTS


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_save_load_moe_layer_single_layer_4x2(bh_2d_mesh_device, tmp_path):
    """Save one MoE layer (4x2 submesh) to disk, load it back, assert metadata and fused-tensor sharing."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load_moe_decoder_layer requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    t0 = time.perf_counter()
    orig = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
    elapsed = time.perf_counter() - t0
    logger.info("prepare_moe_layer_weights (1 MoE layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(orig, DeepSeekV3MoELayerWeights)
    assert orig.q_a_proj.tensor_shape == (3584, 3072)
    assert orig.q_b_proj.tensor_shape == (1536, 12288)
    assert orig.kv_a_proj.tensor_shape == (7168, 576)
    assert orig.o_proj.tensor_shape == (8192, 7168)
    assert orig.gate_mm.tensor_shape == (7168, 256)
    assert orig.gate_bias.shape == (16, 16)
    assert orig.attn_norm.tensor_shape == (1, 7168)
    assert orig.q_norm.tensor_shape == (1, 1536)
    assert orig.kv_norm.tensor_shape == (1, 512)
    assert orig.ffn_norm.tensor_shape == (1, 7168)
    assert orig.kv_b1_proj.tensor_shape == (8192, 512)
    assert orig.kv_b2_proj.tensor_shape == (512, 8192)
    assert orig.shared_gate_proj.tensor_shape == (7168, 256)
    assert orig.shared_up_proj.tensor_shape == (7168, 256)
    assert hasattr(orig, "shared_down_proj")
    assert len(orig.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(orig.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(orig.routed_down_proj) == NUM_ROUTED_EXPERTS
    # Early access to q_ab_kv_a fused_tensor (same as first tensor touched in save_layer)
    logger.info("Early access: touching q_ab_kv_a fused_tensor (orig.q_a_proj.fused_tensor)...")
    q_ab_kv_a_fused = orig.q_a_proj.fused_tensor
    _ = q_ab_kv_a_fused.shape
    logger.info("Early access: got shape {}", q_ab_kv_a_fused.shape)
    save_decoder_layer(
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
    assert (layer_dir / "gate_bias.tensorbin").exists()
    assert (layer_dir / "shared_down_proj.tensorbin").exists()
    experts_dir = layer_dir / "experts"
    for e in range(NUM_ROUTED_EXPERTS):
        expert_dir = experts_dir / f"e_{e:03d}"
        assert (expert_dir / "gate_proj.tensorbin").exists()
        assert (expert_dir / "up_proj.tensorbin").exists()
        assert (expert_dir / "down_proj.tensorbin").exists()

    _deallocate_layer(orig)
    t0 = time.perf_counter()
    layer = load_moe_decoder_layer(tmp_path, submesh, 0)
    elapsed = time.perf_counter() - t0
    logger.info("load_moe_decoder_layer (moe, 4x2 submesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)

    _assert_overlapped_tensors_match(orig.q_a_proj, layer.q_a_proj)
    _assert_overlapped_tensors_match(orig.q_b_proj, layer.q_b_proj)
    _assert_overlapped_tensors_match(orig.kv_a_proj, layer.kv_a_proj)
    _assert_overlapped_tensors_match(orig.o_proj, layer.o_proj)
    _assert_overlapped_tensors_match(orig.gate_mm, layer.gate_mm)
    assert layer.gate_bias.shape == orig.gate_bias.shape
    _assert_overlapped_tensors_match(orig.attn_norm, layer.attn_norm)
    _assert_overlapped_tensors_match(orig.q_norm, layer.q_norm)
    _assert_overlapped_tensors_match(orig.kv_norm, layer.kv_norm)
    _assert_overlapped_tensors_match(orig.ffn_norm, layer.ffn_norm)
    _assert_overlapped_tensors_match(orig.kv_b1_proj, layer.kv_b1_proj)
    _assert_overlapped_tensors_match(orig.kv_b2_proj, layer.kv_b2_proj)
    _assert_overlapped_tensors_match(orig.shared_gate_proj, layer.shared_gate_proj)
    _assert_overlapped_tensors_match(orig.shared_up_proj, layer.shared_up_proj)
    assert layer.shared_down_proj.shape == orig.shared_down_proj.shape
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_down_proj) == NUM_ROUTED_EXPERTS
    for e in range(NUM_ROUTED_EXPERTS):
        assert layer.routed_gate_proj[e].shape == orig.routed_gate_proj[e].shape
        assert layer.routed_up_proj[e].shape == orig.routed_up_proj[e].shape
        assert layer.routed_down_proj[e].shape == orig.routed_down_proj[e].shape

    assert id(layer.shared_gate_proj.fused_tensor) == id(layer.shared_up_proj.fused_tensor)
    _assert_layer_on_device_with_topology(layer)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_load_moe_decoder_layer_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare+save an MoE layer, then load via load_moe_decoder_layer (fast-dispatch for experts is internal) and verify."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load_moe_decoder_layer requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    orig = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
    assert isinstance(orig, DeepSeekV3MoELayerWeights)
    save_decoder_layer(
        orig,
        tmp_path,
        0,
        hf_model_name="test-moe-model-4x2",
        hf_state_dict_name="test-moe-state-dict.safetensors",
        device_mesh_shape=(4, 2),
    )
    _deallocate_layer(orig)

    layer = load_moe_decoder_layer(tmp_path, submesh, 0)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.gate_mm.tensor_shape == (7168, 256)
    assert layer.gate_bias.shape == (16, 16)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.shared_gate_proj.tensor_shape == (7168, 256)
    assert layer.shared_up_proj.tensor_shape == (7168, 256)
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_down_proj) == NUM_ROUTED_EXPERTS
    for e in range(NUM_ROUTED_EXPERTS):
        _assert_on_device(layer.routed_gate_proj[e])
        _assert_on_device(layer.routed_up_proj[e])
        _assert_on_device(layer.routed_down_proj[e])
    _assert_layer_on_device_with_topology(layer)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_embedding_weights_4x2(bh_2d_mesh_device):
    """Prepare embedding weights on 4x2 mesh; verify shape. Tensors stay on host until load_embedding_weights."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = {}
    _add_global_weights(state)
    weights = prepare_embedding_weights(state, submesh)
    assert isinstance(weights, DeepSeekV3EmbeddingLayerWeights)
    assert weights.embedding.shape is not None
    assert weights.embedding.shape == (
        129280,
        7168,
    ), f"Expected embedding shape (129280, 7168), got {weights.embedding.shape}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_lm_head_weights_4x2(bh_2d_mesh_device):
    """Prepare LM head and final norm weights on 4x2 mesh; verify shapes. LM head is vocab-sharded on device (TP=8)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = {}
    _add_global_weights(state)
    weights = prepare_lm_head_weights(state, submesh)
    assert isinstance(weights, DeepSeekV3LMHeadWeights)
    assert weights.lm_head.shape is not None
    assert weights.lm_head.shape == (7168, 16160), f"Expected lm_head shape (7168, 16160), got {weights.lm_head.shape}"
    assert weights.final_norm.shape is not None
    assert weights.final_norm.shape == (1, 7168), f"Expected final_norm shape (1, 7168), got {weights.final_norm.shape}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_save_load_embedding_and_lm_head_weights_4x2(bh_2d_mesh_device, tmp_path):
    """Save embedding and LM head (with final norm) to disk, load both back, verify on device."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = {}
    _add_global_weights(state)
    logger.info("Preparing embedding weights...")
    embedding_weights = prepare_embedding_weights(state, submesh)
    logger.info("Preparing LM head weights...")
    lm_head_weights = prepare_lm_head_weights(state, submesh)
    expected_embedding_shape = embedding_weights.embedding.shape
    expected_lm_shape = lm_head_weights.lm_head.shape
    expected_norm_shape = lm_head_weights.final_norm.shape

    logger.info("Saving embedding...")
    save_embedding_weights(embedding_weights, tmp_path, hf_model_name="test", hf_state_dict_name="test.safetensors")
    logger.info("Saving LM head weights...")
    save_lm_head_weights(
        lm_head_weights,
        tmp_path,
        hf_model_name="test",
        hf_state_dict_name="test.safetensors",
        device_mesh_shape=(4, 2),
    )
    ttnn.deallocate(embedding_weights.embedding, force=True)
    ttnn.deallocate(lm_head_weights.lm_head, force=True)
    ttnn.deallocate(lm_head_weights.final_norm, force=True)

    logger.info("Loading embedding weights...")
    loaded_embedding = load_embedding_weights(tmp_path, submesh)
    assert loaded_embedding.embedding.shape == expected_embedding_shape
    _assert_on_device(loaded_embedding.embedding)

    logger.info("Loading LM head weights...")
    loaded_lm_head = load_lm_head_weights(tmp_path, submesh)
    assert loaded_lm_head.lm_head.shape == expected_lm_shape
    assert loaded_lm_head.final_norm.shape == expected_norm_shape
    _assert_on_device(loaded_lm_head.lm_head)
    _assert_on_device(loaded_lm_head.final_norm)


@pytest.mark.skip(reason="Too slow for CI; use for manual multi-submesh validation")
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
    if not is_slow_dispatch():
        pytest.skip("load_dense_decoder_layer/load_moe_decoder_layer require slow dispatch")
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

    num_layers = 4
    first_k_dense_replace = 3  # layers 0,1,2 dense; layer 3 MoE
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace
        submesh = submeshes[layer_idx]
        bdw = BlitzDecodeWeights(submesh)
        # State for "layer 0" (prepare_*_layer_weights take layer_idx for key lookup)
        state = _layer_state_dict(0, is_moe=is_moe, seed=42 + layer_idx)
        t0 = time.perf_counter()
        if is_moe:
            layer = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
        else:
            layer = prepare_dense_layer_weights(bdw, state, 0)
        elapsed = time.perf_counter() - t0
        logger.info(
            "prepare (layer %d, %s, on submesh %d): {:.3f} s",
            layer_idx,
            "moe" if is_moe else "dense",
            layer_idx,
            elapsed,
        )
        save_decoder_layer(
            layer,
            tmp_path,
            layer_idx,
            hf_model_name="test-4layer-model",
            hf_state_dict_name="test-4layer-state-dict.safetensors",
            device_mesh_shape=(4, 2),
        )
        _deallocate_layer(layer)

    loaded = []
    t0 = time.time()
    for layer_idx in range(num_layers):
        submesh = submeshes[layer_idx]
        if layer_idx >= first_k_dense_replace:
            layer = load_moe_decoder_layer(tmp_path, submesh, layer_idx)
        else:
            layer = load_dense_decoder_layer(tmp_path, submesh, layer_idx)
        loaded.append(layer)
    elapsed = time.time() - t0
    logger.info("load_decoder_layer (4 layers, 4x2 submesh): %s s", elapsed)

    assert isinstance(loaded[0], DeepSeekV3DenseLayerWeights)
    assert isinstance(loaded[1], DeepSeekV3DenseLayerWeights)
    assert isinstance(loaded[2], DeepSeekV3DenseLayerWeights)
    assert isinstance(loaded[3], DeepSeekV3MoELayerWeights)
    for i in range(3):
        assert hasattr(loaded[i], "shared_gate_proj") and loaded[i].shared_gate_proj is not None
        assert hasattr(loaded[i], "routed_gate_proj") and loaded[i].routed_gate_proj is not None
        assert loaded[i].routed_gate_proj.shape is not None
    assert len(loaded[3].routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(loaded[3].routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(loaded[3].routed_down_proj) == NUM_ROUTED_EXPERTS
    assert loaded[3].shared_down_proj is not None
    for i in range(num_layers):
        _assert_layer_on_device_with_topology(loaded[i])
