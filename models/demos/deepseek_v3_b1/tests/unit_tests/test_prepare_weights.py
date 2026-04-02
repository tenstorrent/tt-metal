# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights on 4x2 mesh: prepare_* and TensorCache (CacheConfig) paths.

- Per-component prepare: attention, shared expert, routed expert, dense/MoE layer, embedding, LM head, MTP.
- TensorCache: cold miss then warm hit for the same prepare_* calls.
"""

import time
from dataclasses import fields as dataclass_fields

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights, OverlappedTensor
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions
from models.demos.deepseek_v3_b1.prepare_weights import (
    _MTP_LAYER_IDX,
    CURRENT_TRANSFORM_VERSION,
    AttentionWeights,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    prepare_attention_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_mtp_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
)
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, CacheContext, TensorCache


def _deallocate_layer(layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights) -> None:
    """Deallocate all tensors in a single decoder layer (e.g. after TensorCache cold path)."""
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


_OVERLAPPED_TENSOR_SKIPPED_FIELDS = {"fused_tensor"}


def _assert_overlapped_tensors_match(a: OverlappedTensor, b: OverlappedTensor) -> None:
    """Assert two OverlappedTensors have matching metadata (not fused_tensor identity)."""
    assert a.tensor_shape == b.tensor_shape
    assert a.shard_shape == b.shard_shape
    assert a.dtype == b.dtype
    assert a.tile_shape == b.tile_shape
    assert a.byte_offset == b.byte_offset
    assert a.total_size == b.total_size
    assert _core_range_set_to_tuples(a.core_range_set) == _core_range_set_to_tuples(b.core_range_set)
    checked = {"tensor_shape", "shard_shape", "dtype", "tile_shape", "byte_offset", "total_size", "core_range_set"}
    all_fields = {f.name for f in dataclass_fields(OverlappedTensor)}
    unchecked = all_fields - checked - _OVERLAPPED_TENSOR_SKIPPED_FIELDS
    assert not unchecked, f"OverlappedTensor has new fields not covered by assertion: {unchecked}"


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


def _test_cache_context(mesh_shape: tuple[int, int] = (4, 2)) -> CacheContext:
    return CacheContext(
        schema_version=1,
        hf_model_id="test-model",
        hf_revision="test-rev",
        transform_version=CURRENT_TRANSFORM_VERSION,
        mesh_shape=mesh_shape,
    )


def _deallocate_attention_weights(attn: AttentionWeights) -> None:
    """Deallocate fused tensors and optional gate_bias for attention-only cache tests."""
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
    ):
        ot = getattr(attn, f, None)
        if ot is not None and hasattr(ot, "fused_tensor"):
            fid = id(ot.fused_tensor)
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)
    gm = getattr(attn, "gate_mm", None)
    if gm is not None and hasattr(gm, "fused_tensor"):
        fid = id(gm.fused_tensor)
        if fid not in seen:
            ttnn.deallocate(gm.fused_tensor, force=True)
    gb = getattr(attn, "gate_bias", None)
    if gb is not None:
        ttnn.deallocate(gb, force=True)


# Expected placements for 4x2 mesh (mla_tp=2, moe_tp=8)
_PLACEMENTS_SHARD_NONE_1 = [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]
_PLACEMENTS_SHARD_NONE_0 = [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)]
_PLACEMENTS_SHARD_0_1 = [ttnn.PlacementShard(0), ttnn.PlacementShard(1)]
_PLACEMENTS_REPLICATE = [ttnn.PlacementReplicate()]


# DRAMStreamingMatmul requires gate/up/down expert tensors contiguous per projection (see #40302)
def _assert_moe_layer_routed_experts_dram_contiguous(layer: DeepSeekV3MoELayerWeights) -> None:
    """MoE DRAMStreamingMatmul requires gate/up/down expert tensors contiguous per projection (device only)."""
    MoERoutedExpertWeights(
        routed_gate_proj=layer.routed_gate_proj,
        routed_up_proj=layer.routed_up_proj,
        routed_down_proj=layer.routed_down_proj,
    ).validate_contiguous_dram()


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
        _assert_moe_layer_routed_experts_dram_contiguous(layer)


# HF state dict shapes (out_features, in_features) for linears; full logical for 4x2 mesh. See DEEPSEEK_PREPARE_WEIGHTS_DESIGN_DOC.md §5.
HF_Q_B_FULL_LOGICAL = (LogicalModelDimensions.Q_B_OUT, LogicalModelDimensions.Q_A_DIM)
HF_O_PROJ_FULL_LOGICAL = (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.O_PROJ_OUT)
HF_KV_B_FULL_LOGICAL = (LogicalModelDimensions.KV_B_PROJ_OUT, LogicalModelDimensions.KV_B_LORA_RANK)
HF_SHARED_GATE_UP_FULL_LOGICAL = (LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE)

# Don't use all the experts in tests to avoid taking too long
NUM_ROUTED_EXPERTS_FOR_TESTS = 4


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
            LogicalModelDimensions.Q_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": torch.randn(*q_b_hf, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(
            LogicalModelDimensions.KV_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": torch.randn(
            *kv_b_hf, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(*o_proj_hf, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(
            LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": torch.randn(
            LogicalModelDimensions.Q_A_DIM, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": torch.randn(
            LogicalModelDimensions.KV_B_LORA_RANK, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": torch.randn(
            LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16
        ),
    }
    if is_moe:
        state[f"model.layers.{layer_idx}.mlp.gate.weight"] = torch.randn(
            LogicalModelDimensions.GATE_NUM_INDICES,
            LogicalModelDimensions.HIDDEN_SIZE,
            generator=g,
            dtype=torch.bfloat16,
        )
        state[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = torch.randn(
            LogicalModelDimensions.GATE_NUM_INDICES, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        # shared down: HF (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE*tp) full logical
        shared_down_rows = LogicalModelDimensions.HIDDEN_SIZE
        shared_down_cols = shared_hf[0]  # 2048 for 4x2
        state[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(
            shared_down_rows, shared_down_cols, generator=g, dtype=torch.bfloat16
        )
        for e in range(NUM_ROUTED_EXPERTS_FOR_TESTS):
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                LogicalModelDimensions.MOE_INTERMEDIATE_SIZE,
                LogicalModelDimensions.HIDDEN_SIZE,
                generator=g,
                dtype=torch.bfloat16,
            )
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                LogicalModelDimensions.MOE_INTERMEDIATE_SIZE,
                LogicalModelDimensions.HIDDEN_SIZE,
                generator=g,
                dtype=torch.bfloat16,
            )
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                LogicalModelDimensions.HIDDEN_SIZE,
                LogicalModelDimensions.MOE_INTERMEDIATE_SIZE,
                generator=g,
                dtype=torch.bfloat16,
            )
    else:
        # Dense MLP: HF (out, in) gate/up (INTERMEDIATE_SIZE, HIDDEN_SIZE), down (HIDDEN_SIZE, INTERMEDIATE_SIZE)
        state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.randn(
            LogicalModelDimensions.INTERMEDIATE_SIZE,
            LogicalModelDimensions.HIDDEN_SIZE,
            generator=g,
            dtype=torch.bfloat16,
        )
        state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.randn(
            LogicalModelDimensions.INTERMEDIATE_SIZE,
            LogicalModelDimensions.HIDDEN_SIZE,
            generator=g,
            dtype=torch.bfloat16,
        )
        state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.randn(
            LogicalModelDimensions.HIDDEN_SIZE,
            LogicalModelDimensions.INTERMEDIATE_SIZE,
            generator=g,
            dtype=torch.bfloat16,
        )
    return state


def _add_global_weights(state: dict[str, torch.Tensor], seed: int = 42) -> None:
    """Add embedding, final norm, and lm_head to state (in place)."""
    g = torch.Generator().manual_seed(seed)
    state["model.embed_tokens.weight"] = torch.randn(
        LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16
    )
    state["model.norm.weight"] = torch.randn(LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(
        LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE, generator=g, dtype=torch.bfloat16
    )


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
    assert attn.q_b_proj.tensor_shape == (LogicalModelDimensions.Q_A_DIM, 12288)
    assert attn.kv_a_proj.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.KV_A_DIM)
    assert attn.o_proj.tensor_shape == (8192, LogicalModelDimensions.HIDDEN_SIZE)
    assert attn.attn_norm.tensor_shape == (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert attn.kv_b1_proj.tensor_shape == (8192, LogicalModelDimensions.KV_B_LORA_RANK)
    assert attn.kv_b2_proj.tensor_shape == (LogicalModelDimensions.KV_B_LORA_RANK, 8192)


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
    assert attn.gate_mm.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.GATE_NUM_INDICES)
    assert attn.gate_bias is not None
    assert attn.gate_bias.shape == (16, 16)
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    assert attn.o_proj.tensor_shape == (8192, LogicalModelDimensions.HIDDEN_SIZE)


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
    assert shared.shared_gate_proj.tensor_shape == (
        LogicalModelDimensions.HIDDEN_SIZE,
        LogicalModelDimensions.GATE_NUM_INDICES,
    )
    assert shared.shared_up_proj.tensor_shape == (
        LogicalModelDimensions.HIDDEN_SIZE,
        LogicalModelDimensions.GATE_NUM_INDICES,
    )
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
    routed = prepare_routed_expert_weights(
        bdw,
        state,
        0,
        is_moe=True,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        move_to_device=True,
    )
    assert isinstance(routed, MoERoutedExpertWeights)
    assert len(routed.routed_gate_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(routed.routed_up_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(routed.routed_down_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    routed.validate_contiguous_dram()


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
    assert layer.q_b_proj.tensor_shape == (LogicalModelDimensions.Q_A_DIM, 12288)
    assert layer.kv_a_proj.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.KV_A_DIM)
    assert layer.o_proj.tensor_shape == (8192, LogicalModelDimensions.HIDDEN_SIZE)
    assert layer.attn_norm.tensor_shape == (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert layer.q_norm.tensor_shape == (1, LogicalModelDimensions.Q_A_DIM)
    assert layer.kv_norm.tensor_shape == (1, LogicalModelDimensions.KV_B_LORA_RANK)
    assert layer.ffn_norm.tensor_shape == (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert layer.kv_b1_proj.tensor_shape == (8192, LogicalModelDimensions.KV_B_LORA_RANK)
    assert layer.kv_b2_proj.tensor_shape == (LogicalModelDimensions.KV_B_LORA_RANK, 8192)
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
def test_prepare_moe_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one MoE layer on 4x2 mesh; verify type and shapes (MLA TP=2, MoE TP=8)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    logger.info(f"State dict prepared")
    t0 = time.perf_counter()
    logger.info(f"Preparing weights...")
    layer = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS)
    logger.info(f"Weights prepared")
    elapsed = time.perf_counter() - t0
    logger.info("prepare_moe_layer_weights (1 MoE layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (LogicalModelDimensions.Q_A_DIM, 12288)
    assert layer.kv_a_proj.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.KV_A_DIM)
    assert layer.o_proj.tensor_shape == (8192, LogicalModelDimensions.HIDDEN_SIZE)
    assert layer.gate_mm.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.GATE_NUM_INDICES)
    assert layer.gate_bias.shape == (16, 16)
    assert layer.attn_norm.tensor_shape == (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert layer.q_norm.tensor_shape == (1, LogicalModelDimensions.Q_A_DIM)
    assert layer.kv_norm.tensor_shape == (1, LogicalModelDimensions.KV_B_LORA_RANK)
    assert layer.ffn_norm.tensor_shape == (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert layer.kv_b1_proj.tensor_shape == (8192, LogicalModelDimensions.KV_B_LORA_RANK)
    assert layer.kv_b2_proj.tensor_shape == (LogicalModelDimensions.KV_B_LORA_RANK, 8192)
    assert layer.shared_gate_proj.tensor_shape == (
        LogicalModelDimensions.HIDDEN_SIZE,
        LogicalModelDimensions.GATE_NUM_INDICES,
    )
    assert layer.shared_up_proj.tensor_shape == (
        LogicalModelDimensions.HIDDEN_SIZE,
        LogicalModelDimensions.GATE_NUM_INDICES,
    )
    assert hasattr(layer, "shared_down_proj")
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(layer.routed_up_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(layer.routed_down_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_embedding_weights_4x2(bh_2d_mesh_device):
    """Prepare embedding weights on 4x2 mesh; verify shape."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = {}
    _add_global_weights(state)
    weights = prepare_embedding_weights(state, submesh)
    assert isinstance(weights, DeepSeekV3EmbeddingLayerWeights)
    assert weights.embedding.shape is not None
    assert weights.embedding.shape == (
        LogicalModelDimensions.VOCAB_SIZE,
        LogicalModelDimensions.HIDDEN_SIZE,
    ), f"Expected embedding shape ({LogicalModelDimensions.VOCAB_SIZE}, {LogicalModelDimensions.HIDDEN_SIZE}), got {weights.embedding.shape}"


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
    assert weights.lm_head.shape == (
        LogicalModelDimensions.HIDDEN_SIZE,
        16160,
    ), f"Expected lm_head shape ({LogicalModelDimensions.HIDDEN_SIZE}, 16160), got {weights.lm_head.shape}"
    assert weights.final_norm.shape is not None
    assert weights.final_norm.shape == (
        1,
        LogicalModelDimensions.HIDDEN_SIZE,
    ), f"Expected final_norm shape (1, {LogicalModelDimensions.HIDDEN_SIZE}), got {weights.final_norm.shape}"


def _mtp_state_dict(mtp_layer_idx: int = _MTP_LAYER_IDX, seed: int = 44) -> dict[str, torch.Tensor]:
    """Build a synthetic state dict with only the lightweight MTP projection/norm tensors."""
    g = torch.Generator().manual_seed(seed + 1000)
    dtype = torch.bfloat16
    H = LogicalModelDimensions.HIDDEN_SIZE
    return {
        f"model.layers.{mtp_layer_idx}.hnorm.weight": torch.randn(H, generator=g, dtype=dtype),
        f"model.layers.{mtp_layer_idx}.enorm.weight": torch.randn(H, generator=g, dtype=dtype),
        f"model.layers.{mtp_layer_idx}.eh_proj.weight": torch.randn(H, 2 * H, generator=g, dtype=dtype),
    }


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_mtp_weights_4x2(bh_2d_mesh_device):
    """Prepare MTP weights on 4x2 mesh; verify type and shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _mtp_state_dict()
    t0 = time.perf_counter()
    weights = prepare_mtp_weights(state, submesh)
    elapsed = time.perf_counter() - t0
    logger.info("prepare_mtp_weights (4x2 mesh): {:.3f} s", elapsed)
    H = LogicalModelDimensions.HIDDEN_SIZE
    assert isinstance(weights, DeepSeekV3MTPWeights)
    assert weights.h_gamma.shape == (1, H)
    assert weights.e_gamma.shape == (1, H)
    assert weights.eh_projection.shape == (2 * H, H)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_embedding_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare embedding weights via TensorCache on 4x2 mesh: cold miss then warm hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = {}
    _add_global_weights(state)

    weights = prepare_embedding_weights(state, submesh, cache_config=cache_config)
    assert isinstance(weights, DeepSeekV3EmbeddingLayerWeights)
    expected_shape = (LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE)
    assert weights.embedding.shape == expected_shape, f"Expected {expected_shape}, got {weights.embedding.shape}"

    ttnn.deallocate(weights.embedding, force=True)

    weights_hit = prepare_embedding_weights(state, submesh, cache_config=cache_config)
    assert weights_hit.embedding.shape == expected_shape

    objects_dir = cache_config.cache.local_root / "objects"
    assert objects_dir.exists()
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert len(artifact_dirs) >= 1, f"Expected at least 1 cached artifact, found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_lm_head_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare LM head + final norm via TensorCache on 4x2 mesh: cold miss then warm hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = {}
    _add_global_weights(state)

    weights = prepare_lm_head_weights(state, submesh, cache_config=cache_config)
    assert isinstance(weights, DeepSeekV3LMHeadWeights)
    expected_lm = (LogicalModelDimensions.HIDDEN_SIZE, 16160)
    expected_norm = (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert weights.lm_head.shape == expected_lm, f"Expected lm_head {expected_lm}, got {weights.lm_head.shape}"
    assert (
        weights.final_norm.shape == expected_norm
    ), f"Expected final_norm {expected_norm}, got {weights.final_norm.shape}"

    ttnn.deallocate(weights.lm_head, force=True)
    ttnn.deallocate(weights.final_norm, force=True)

    weights_hit = prepare_lm_head_weights(state, submesh, cache_config=cache_config)
    assert weights_hit.lm_head.shape == expected_lm
    assert weights_hit.final_norm.shape == expected_norm

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert len(artifact_dirs) >= 2, f"Expected at least 2 cached artifacts (lm_head + norm), found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_attention_weights_with_cache_dense_4x2(bh_2d_mesh_device, tmp_path):
    """Attention fusion groups (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms) via TensorCache: miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)

    attn = prepare_attention_weights(bdw, state, 0, is_moe=False, cache_config=cache_config)
    assert attn.gate_mm is None
    assert attn.q_a_proj.tensor_shape == (3584, 3072)

    _deallocate_attention_weights(attn)

    attn_hit = prepare_attention_weights(bdw, state, 0, is_moe=False, cache_config=cache_config)
    assert attn_hit.gate_mm is None
    assert attn_hit.q_a_proj.tensor_shape == attn.q_a_proj.tensor_shape

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert (
        len(artifact_dirs) >= 3
    ), f"Expected 3 fusion artifacts (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms), found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_attention_weights_with_cache_moe_4x2(bh_2d_mesh_device, tmp_path):
    """Attention fusion groups + gate_bias via TensorCache on MoE layer: miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)

    attn = prepare_attention_weights(bdw, state, 0, is_moe=True, cache_config=cache_config)
    assert attn.gate_mm is not None
    assert attn.gate_bias is not None

    _deallocate_attention_weights(attn)

    attn_hit = prepare_attention_weights(bdw, state, 0, is_moe=True, cache_config=cache_config)
    assert attn_hit.gate_mm is not None
    assert attn_hit.gate_bias.shape == (16, 16)

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert len(artifact_dirs) >= 4, f"Expected 3 fusion artifacts + gate_bias, found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_shared_expert_weights_with_cache_dense_4x2(bh_2d_mesh_device, tmp_path):
    """gate_up fusion group via TensorCache (dense path): miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)

    shared = prepare_shared_expert_weights(bdw, state, 0, is_moe=False, cache_config=cache_config)
    assert shared.shared_gate_proj.tensor_shape is not None

    ttnn.deallocate(shared.shared_gate_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_up_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_down_proj, force=True)

    shared_hit = prepare_shared_expert_weights(bdw, state, 0, is_moe=False, cache_config=cache_config)
    assert shared_hit.shared_gate_proj.tensor_shape == shared.shared_gate_proj.tensor_shape

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert len(artifact_dirs) >= 2, f"Expected gate_up + shared_down_proj artifacts, found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_shared_expert_weights_with_cache_moe_4x2(bh_2d_mesh_device, tmp_path):
    """gate_up fusion group via TensorCache (MoE path): miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)

    shared = prepare_shared_expert_weights(bdw, state, 0, is_moe=True, cache_config=cache_config)
    assert shared.shared_gate_proj.tensor_shape is not None

    ttnn.deallocate(shared.shared_gate_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_up_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_down_proj, force=True)

    shared_hit = prepare_shared_expert_weights(bdw, state, 0, is_moe=True, cache_config=cache_config)
    assert shared_hit.shared_gate_proj.tensor_shape == shared.shared_gate_proj.tensor_shape

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert len(artifact_dirs) >= 2, f"Expected gate_up + shared_down_proj artifacts, found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_with_cache_dense_4x2(bh_2d_mesh_device, tmp_path):
    """Dense MLP routed projections (stacked on mesh) via TensorCache: miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)

    routed = prepare_routed_expert_weights(bdw, state, 0, is_moe=False, cache_config=cache_config)
    assert isinstance(routed, DenseRoutedExpertWeights)

    ttnn.deallocate(routed.routed_gate_proj, force=True)
    ttnn.deallocate(routed.routed_up_proj, force=True)
    ttnn.deallocate(routed.routed_down_proj, force=True)

    routed_hit = prepare_routed_expert_weights(bdw, state, 0, is_moe=False, cache_config=cache_config)
    assert isinstance(routed_hit, DenseRoutedExpertWeights)

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert len(artifact_dirs) >= 3, f"Expected 3 stacked routed artifacts (gate/up/down), found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_with_cache_moe_4x2(bh_2d_mesh_device, tmp_path):
    """MoE routed experts (per-expert DRAM) via TensorCache: miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)

    routed = prepare_routed_expert_weights(
        bdw,
        state,
        0,
        is_moe=True,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        cache_config=cache_config,
    )
    assert isinstance(routed, MoERoutedExpertWeights)
    assert len(routed.routed_gate_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS

    for t in routed.routed_gate_proj + routed.routed_up_proj + routed.routed_down_proj:
        ttnn.deallocate(t, force=True)

    routed_hit = prepare_routed_expert_weights(
        bdw,
        state,
        0,
        is_moe=True,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        cache_config=cache_config,
    )
    assert isinstance(routed_hit, MoERoutedExpertWeights)

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert (
        len(artifact_dirs) >= NUM_ROUTED_EXPERTS_FOR_TESTS * 3
    ), f"Expected {NUM_ROUTED_EXPERTS_FOR_TESTS * 3} per-expert routed artifacts, found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_dense_layer_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Full dense layer via TensorCache: attention + gate_up + shared_down + routed; miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)

    layer = prepare_dense_layer_weights(bdw, state, 0, cache_config=cache_config)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)

    _deallocate_layer(layer)

    layer_hit = prepare_dense_layer_weights(bdw, state, 0, cache_config=cache_config)
    assert isinstance(layer_hit, DeepSeekV3DenseLayerWeights)
    assert layer_hit.q_a_proj.tensor_shape == layer.q_a_proj.tensor_shape

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert (
        len(artifact_dirs) >= 8
    ), f"Expected 3 attention fusion + gate_up + shared_down + 3 routed stacked (8), found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_layer_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare MoE layer via TensorCache: fusion + gate_bias + gate_up + shared_down + routed experts."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    layer_idx = 3
    state = _layer_state_dict(layer_idx, is_moe=True)
    bdw = BlitzDecodeWeights(submesh)

    weights = prepare_moe_layer_weights(
        bdw,
        state,
        layer_idx,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        cache_config=cache_config,
    )
    assert isinstance(weights, DeepSeekV3MoELayerWeights)
    expected_gate_bias = (16, 16)
    assert (
        weights.gate_bias.shape == expected_gate_bias
    ), f"Expected gate_bias {expected_gate_bias}, got {weights.gate_bias.shape}"

    _deallocate_layer(weights)

    weights_hit = prepare_moe_layer_weights(
        bdw,
        state,
        layer_idx,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        cache_config=cache_config,
    )
    assert weights_hit.gate_bias.shape == expected_gate_bias

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    n_r = NUM_ROUTED_EXPERTS_FOR_TESTS
    # 3 attention fusion + gate_bias + gate_up + shared_down + n_r * 3 routed = 6 + n_r * 3
    assert len(artifact_dirs) >= 6 + n_r * 3, (
        f"Expected 3 attn + gate_bias + gate_up + shared_down + {n_r * 3} routed ({6 + n_r * 3}), "
        f"found {len(artifact_dirs)}"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_mtp_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare MTP weights via TensorCache on 4x2 mesh: cold miss then warm hit for h/e gamma, eh_proj."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _mtp_state_dict()
    H = LogicalModelDimensions.HIDDEN_SIZE

    weights = prepare_mtp_weights(state, submesh, cache_config=cache_config)
    assert isinstance(weights, DeepSeekV3MTPWeights)
    assert weights.h_gamma.shape == (1, H)
    assert weights.e_gamma.shape == (1, H)
    assert weights.eh_projection.shape == (2 * H, H)

    expected_shapes = {
        "h_gamma": weights.h_gamma.shape,
        "e_gamma": weights.e_gamma.shape,
        "eh_projection": weights.eh_projection.shape,
    }

    ttnn.deallocate(weights.h_gamma, force=True)
    ttnn.deallocate(weights.e_gamma, force=True)
    ttnn.deallocate(weights.eh_projection, force=True)

    weights_hit = prepare_mtp_weights(state, submesh, cache_config=cache_config)
    assert weights_hit.h_gamma.shape == expected_shapes["h_gamma"]
    assert weights_hit.e_gamma.shape == expected_shapes["e_gamma"]
    assert weights_hit.eh_projection.shape == expected_shapes["eh_projection"]

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert (
        len(artifact_dirs) >= 3
    ), f"Expected at least 3 cached artifacts (h/e gamma, eh_proj), found {len(artifact_dirs)}"
