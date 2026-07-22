# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights on 4x2 mesh: prepare_* and TensorCache (CacheConfig) paths.

- Per-component prepare: attention, shared expert, routed expert, dense/MoE layer, embedding, LM head, MTP.
- TensorCache: cold miss then warm hit for the same prepare_* calls.
"""

import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from conftest import requires_hybrid_allocator
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, bfp4_tile_byte_count
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions
from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlappedTensor
from models.demos.deepseek_v3_b1.weights.prepare import (
    _MTP_LAYER_IDX,
    AttentionWeights,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    DeepSeekV3SpecWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    SharedExpertWeights,
    create_gate_indices_tensor,
    prepare_attention_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_moe_routed_experts_bspm,
    prepare_mtp_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
    prepare_spec_weights,
)


def _deallocate_layer(layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights) -> None:
    """Deallocate all tensors in a single decoder layer (e.g. after TensorCache cold path).

    ``gate_mm`` is listed explicitly because MoE layers pack it into its own
    per-core fusion artefact (see ``MERGED_TP4_GATE_SPEC``), so its
    ``fused_tensor`` is distinct from the main attention buffer and is not
    freed transitively by deallocating ``o_proj``.
    """
    seen: set[int] = set()
    for f in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "o_proj",
        "gate_mm",
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
            fid = ot.fused_tensor.tensor_id
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)
    ttnn.deallocate(layer.shared_down_proj, force=True)
    if isinstance(layer, DeepSeekV3MoELayerWeights):
        ttnn.deallocate(layer.gate_bias, force=True)
    for projection in (layer.routed_gate_proj, layer.routed_up_proj, layer.routed_down_proj):
        for t in projection:
            # TP8 path stores CompressedTensors in these lists (data + assignment per CT);
            # the legacy uniform-BFP4 path stores plain ttnn.Tensors.
            if isinstance(t, CompressedTensor):
                t.deallocate(force=True)
            else:
                ttnn.deallocate(t, force=True)


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
    assert a.total_size == b.total_size
    assert _core_range_set_to_tuples(a.core_range_set) == _core_range_set_to_tuples(b.core_range_set)


_ATTENTION_OVERLAPPED_FIELDS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj",
    "o_proj",
    "gate_mm",
    "attn_norm",
    "q_norm",
    "kv_norm",
    "ffn_norm",
    "kv_b1_proj",
    "kv_b2_proj",
)
_SHARED_EXPERT_OVERLAPPED_FIELDS = ("shared_gate_proj", "shared_up_proj")


def _assert_attention_metadata_matches(cold: AttentionWeights, warm: AttentionWeights) -> None:
    """Assert cold and warm cache runs produce identical OverlappedTensor metadata for attention."""
    for field in _ATTENTION_OVERLAPPED_FIELDS:
        a = getattr(cold, field, None)
        b = getattr(warm, field, None)
        if a is None and b is None:
            continue
        assert a is not None and b is not None, f"{field}: one is None, other is not"
        _assert_overlapped_tensors_match(a, b)


def _assert_shared_expert_metadata_matches(cold: SharedExpertWeights, warm: SharedExpertWeights) -> None:
    """Assert cold and warm cache runs produce identical OverlappedTensor metadata for shared expert."""
    for field in _SHARED_EXPERT_OVERLAPPED_FIELDS:
        _assert_overlapped_tensors_match(getattr(cold, field), getattr(warm, field))


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
            fid = ot.fused_tensor.tensor_id
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)
    gm = getattr(attn, "gate_mm", None)
    if gm is not None and hasattr(gm, "fused_tensor"):
        fid = gm.fused_tensor.tensor_id
        if fid not in seen:
            ttnn.deallocate(gm.fused_tensor, force=True)
    gb = getattr(attn, "gate_bias", None)
    if gb is not None:
        ttnn.deallocate(gb, force=True)


# Expected placements for 4x2 mesh (mla_tp=2, moe_tp=8)
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
    fid = layer.q_a_proj.fused_tensor.tensor_id
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.q_a_proj.fused_tensor, _PLACEMENTS_SHARD_0_1)
    # o_proj_gate_mm_norms
    _assert_on_device(layer.o_proj.fused_tensor)
    fid = layer.o_proj.fused_tensor.tensor_id
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.o_proj.fused_tensor, _PLACEMENTS_SHARD_0_1)
    # kv_b12
    _assert_on_device(layer.kv_b1_proj.fused_tensor)
    fid = layer.kv_b1_proj.fused_tensor.tensor_id
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.kv_b1_proj.fused_tensor, _PLACEMENTS_SHARD_0_1)
    # gate_up
    _assert_on_device(layer.shared_gate_proj.fused_tensor)
    fid = layer.shared_gate_proj.fused_tensor.tensor_id
    if fid not in seen_fused:
        seen_fused.add(fid)
        _assert_topology(layer.shared_gate_proj.fused_tensor, _PLACEMENTS_SHARD_0_1)
    # Standalone: shared_down_proj
    _assert_on_device(layer.shared_down_proj)
    _assert_topology(layer.shared_down_proj, _PLACEMENTS_SHARD_0_1)
    # Routed experts
    # Routed experts are now per-projection lists for both dense and MoE. Dense always
    # uses the TP8 CompressedTensor path (2D-sharded placements); MoE in tests still
    # goes through the legacy ttnn.Tensor path (replicate per expert) since these
    # tests don't pass ``compressed_tp8=True``.
    if isinstance(layer, DeepSeekV3DenseLayerWeights):
        expected_placements = _PLACEMENTS_SHARD_0_1
    else:
        assert isinstance(layer, DeepSeekV3MoELayerWeights)
        _assert_topology(layer.gate_bias, _PLACEMENTS_REPLICATE)
        expected_placements = _PLACEMENTS_REPLICATE

    def _underlying(t):
        return t.get_data_tensor() if isinstance(t, CompressedTensor) else t

    for e in range(len(layer.routed_gate_proj)):
        for t in (layer.routed_gate_proj[e], layer.routed_up_proj[e], layer.routed_down_proj[e]):
            data = _underlying(t)
            _assert_on_device(data)
            _assert_topology(data, expected_placements)
    if isinstance(layer, DeepSeekV3MoELayerWeights):
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


def test_compressed_tensor_target_transform_version_invalidates_cache():
    """Bumping transform_version on CompressedTensorTarget produces a distinct artifact ID."""
    from models.demos.deepseek_v3_b1.weights.cache import (
        BspmVariant,
        CacheContext,
        CompressedTensorTarget,
        SourceTensorSelection,
    )
    from models.demos.deepseek_v3_b1.weights.cache.fingerprint import compute_artifact_id

    ctx = CacheContext(schema_version=1, hf_model_id="test", hf_revision="r0", mesh_shape=(1, 1))
    source = SourceTensorSelection(names=("layer.weight",))

    tgt_v3 = CompressedTensorTarget(
        name="gate_proj",
        K=64,
        N_padded=64,
        num_banks=8,
        bspm_variant=BspmVariant.B,
        bspm_budget=3.5,
        transform_version=3,
    )
    tgt_v4 = CompressedTensorTarget(
        name="gate_proj",
        K=64,
        N_padded=64,
        num_banks=8,
        bspm_variant=BspmVariant.B,
        bspm_budget=3.5,
        transform_version=4,
    )

    id_v3 = compute_artifact_id(ctx.fingerprint(source=source, target=tgt_v3))
    id_v4 = compute_artifact_id(ctx.fingerprint(source=source, target=tgt_v4))
    assert id_v3 != id_v4, "Bumping transform_version must produce a different artifact ID"


def test_compressed_tensor_target_assignment_hash_invalidates_cache():
    """Different assignment_hash values in CompressedTensorTarget produce distinct artifact IDs."""
    from models.demos.deepseek_v3_b1.weights.cache import (
        BspmVariant,
        CacheContext,
        CompressedTensorTarget,
        SourceTensorSelection,
    )
    from models.demos.deepseek_v3_b1.weights.cache.fingerprint import compute_artifact_id

    ctx = CacheContext(schema_version=1, hf_model_id="test", hf_revision="r0", mesh_shape=(1, 1))
    source = SourceTensorSelection(names=("layer.weight",))

    tgt_a = CompressedTensorTarget(
        name="gate_proj",
        K=64,
        N_padded=64,
        num_banks=8,
        bspm_variant=BspmVariant.B,
        bspm_budget=3.5,
        assignment_hash="aabbccdd00001111",
    )
    tgt_b = CompressedTensorTarget(
        name="gate_proj",
        K=64,
        N_padded=64,
        num_banks=8,
        bspm_variant=BspmVariant.B,
        bspm_budget=3.5,
        assignment_hash="ffffffffffffffff",
    )

    id_a = compute_artifact_id(ctx.fingerprint(source=source, target=tgt_a))
    id_b = compute_artifact_id(ctx.fingerprint(source=source, target=tgt_b))
    assert id_a != id_b, "Different assignment_hash values must produce distinct artifact IDs"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_attention_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare attention weights only for a dense layer on 4x2 mesh; verify shapes and fusion group sharing."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)

    attn = prepare_attention_weights(submesh, state, 0, is_moe=False)
    assert attn.gate_mm is None
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    assert attn.q_b_proj.tensor_shape == (LogicalModelDimensions.Q_A_DIM, 12288)
    assert attn.kv_a_proj.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.KV_A_DIM)
    # o_proj is TP4-shuffle-packed to (8192, 2 * HIDDEN_SIZE); see pack_o_proj_weights_tp4_shuffled.
    assert attn.o_proj.tensor_shape == (8192, 2 * LogicalModelDimensions.HIDDEN_SIZE)
    assert attn.attn_norm.tensor_shape == (1, LogicalModelDimensions.HIDDEN_SIZE)
    assert attn.kv_b1_proj.tensor_shape == (8192, LogicalModelDimensions.KV_B_LORA_RANK)
    assert attn.kv_b2_proj.tensor_shape == (LogicalModelDimensions.KV_B_LORA_RANK, 8192)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_attention_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare attention weights only for an MoE layer on 4x2 mesh; verify shapes and gate_mm present."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)

    attn = prepare_attention_weights(submesh, state, 0, is_moe=True)
    assert attn.gate_mm is not None
    assert attn.gate_mm.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.GATE_NUM_INDICES)
    assert attn.gate_bias is not None
    assert attn.gate_bias.shape == (16, 16)
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    # o_proj is TP4-shuffle-packed to (8192, 2 * HIDDEN_SIZE); see pack_o_proj_weights_tp4_shuffled.
    assert attn.o_proj.tensor_shape == (8192, 2 * LogicalModelDimensions.HIDDEN_SIZE)


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

    shared = prepare_shared_expert_weights(submesh, state, 0, is_moe=False)
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

    shared = prepare_shared_expert_weights(submesh, state, 0, is_moe=True)
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

    # move_to_device=True: dense routed now goes through prepare_dense_routed_experts_compressed_tp8 →
    # get_or_create_bspm_expert_tp8 → expert_dram_memory_config(dev, ...), which requires a non-None device.
    routed = prepare_routed_expert_weights(submesh, state, 0, is_moe=False, move_to_device=True)
    assert isinstance(routed, DenseRoutedExpertWeights)
    # Dense MLP routed portion is sliced into _dn_num_routed=8 TP8-sharded CompressedTensors
    # per projection — one per chunk, identical per-device shape to a MoE routed expert,
    # so the kernel iterates num_active_experts=8 the same way it does for MoE.
    expected_chunks = (
        LogicalModelDimensions.INTERMEDIATE_SIZE - LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    ) // LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    assert len(routed.routed_gate_proj) == expected_chunks
    assert len(routed.routed_up_proj) == expected_chunks
    assert len(routed.routed_down_proj) == expected_chunks
    for ct in (*routed.routed_gate_proj, *routed.routed_up_proj, *routed.routed_down_proj):
        assert isinstance(ct, CompressedTensor)


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

    routed = prepare_routed_expert_weights(
        submesh,
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
@requires_hybrid_allocator
def test_prepare_dense_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one dense layer on 4x2 mesh; verify type and shapes (MLA TP=2)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)

    t0 = time.perf_counter()
    # move_to_device=True: dense routed CompressedTensor cache reconstruct needs a non-None device
    # (see test_prepare_routed_expert_weights_dense_4x2 for details).
    layer = prepare_dense_layer_weights(submesh, state, 0, move_to_device=True)
    elapsed = time.perf_counter() - t0
    logger.info("prepare_dense_layer_weights (1 dense layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (LogicalModelDimensions.Q_A_DIM, 12288)
    assert layer.kv_a_proj.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.KV_A_DIM)
    # o_proj is TP4-shuffle-packed to (8192, 2 * HIDDEN_SIZE); see pack_o_proj_weights_tp4_shuffled.
    assert layer.o_proj.tensor_shape == (8192, 2 * LogicalModelDimensions.HIDDEN_SIZE)
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
@requires_hybrid_allocator
def test_prepare_moe_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one MoE layer on 4x2 mesh; verify type and shapes (MLA TP=2, MoE TP=8)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)

    logger.info(f"State dict prepared")
    t0 = time.perf_counter()
    logger.info(f"Preparing weights...")
    layer = prepare_moe_layer_weights(submesh, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS)
    logger.info(f"Weights prepared")
    elapsed = time.perf_counter() - t0
    logger.info("prepare_moe_layer_weights (1 MoE layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (LogicalModelDimensions.Q_A_DIM, 12288)
    assert layer.kv_a_proj.tensor_shape == (LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.KV_A_DIM)
    # o_proj is TP4-shuffle-packed to (8192, 2 * HIDDEN_SIZE); see pack_o_proj_weights_tp4_shuffled.
    assert layer.o_proj.tensor_shape == (8192, 2 * LogicalModelDimensions.HIDDEN_SIZE)
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


# =============================================================================
# SRAM-resident routed expert tests (sram_expert_ids + compressed_tp8 paths)
# =============================================================================

# Bit-15 marks an expert ID as SRAM-resident; the low 15 bits carry the slot index.
# See create_gate_indices_tensor docstring for the wire-encoding contract that
# build_sram_routed_proj_cts and the kernel-side filter both depend on.
_SRAM_BIT = 1 << 15


def _read_gate_indices_back(submesh, indices_tensor: ttnn.Tensor) -> torch.Tensor:
    """Read a ReplicateTensorToMesh-replicated 16x16 uint16 indices tensor back as a flat (256,) long.

    Undoes the (reshape → transpose) layout from create_gate_indices_tensor so position e of the
    returned tensor holds the encoding for global expert id e.
    """
    host = ttnn.to_torch(
        indices_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=(submesh.shape[0], submesh.shape[1]), dims=(0, 1)),
    )
    # Replicated across all devices → every (16, 16) slice is identical; take the first.
    one_replica = host[:16, :16]
    # Undo `torch.transpose(indices, 0, 1)` then `reshape(16, 16)` from create_gate_indices_tensor.
    return one_replica.transpose(0, 1).reshape(-1).to(torch.int64)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_create_gate_indices_tensor_sram_encoding_4x2(bh_2d_mesh_device):
    """SRAM bit-15 slot encoding round-trips correctly; DRAM eids stay identity."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    sender_core = ttnn.CoreCoord(12, 9)
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)

    # Default (no SRAM expert IDs): identity arange [0..255], no bit-15 set anywhere.
    default_t = create_gate_indices_tensor(submesh, sender_grid, mesh_mapper=mesh_mapper)
    default_flat = _read_gate_indices_back(submesh, default_t)
    assert torch.equal(default_flat, torch.arange(256, dtype=torch.int64))
    assert ((default_flat & _SRAM_BIT) == 0).all(), "no SRAM bit should be set in default encoding"
    ttnn.deallocate(default_t, force=True)

    # SRAM expert IDs: slot s ↔ eid sram_expert_ids[s]; bit-15 set + slot in low bits;
    # all other eids remain identity (DRAM).
    sram_expert_ids = [5, 100, 200]
    sram_t = create_gate_indices_tensor(submesh, sender_grid, sram_expert_ids=sram_expert_ids, mesh_mapper=mesh_mapper)
    sram_flat = _read_gate_indices_back(submesh, sram_t)
    for slot, eid in enumerate(sram_expert_ids):
        expected = _SRAM_BIT | slot
        assert (
            sram_flat[eid].item() == expected
        ), f"eid={eid} slot={slot}: got {sram_flat[eid].item():#x}, want {expected:#x}"
    dram_mask = torch.ones(256, dtype=torch.bool)
    dram_mask[sram_expert_ids] = False
    assert torch.equal(sram_flat[dram_mask], torch.arange(256, dtype=torch.int64)[dram_mask])
    ttnn.deallocate(sram_t, force=True)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_create_gate_indices_tensor_sram_validation_4x2(bh_2d_mesh_device):
    """create_gate_indices_tensor rejects malformed sram_expert_ids before any device upload."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    sender_core = ttnn.CoreCoord(12, 9)
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)

    with pytest.raises(AssertionError, match="out-of-range"):
        create_gate_indices_tensor(submesh, sender_grid, sram_expert_ids=[256], mesh_mapper=mesh_mapper)
    with pytest.raises(AssertionError, match="duplicates"):
        create_gate_indices_tensor(submesh, sender_grid, sram_expert_ids=[5, 5], mesh_mapper=mesh_mapper)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_moe_layer_weights_with_sram_expert_ids_4x2(bh_2d_mesh_device):
    """sram_expert_ids populates sram_*_proj per slot; DRAM expert list stays at num_routed_experts."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)

    sram_expert_ids = [0, 1]
    layer = prepare_moe_layer_weights(
        submesh,
        state,
        0,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        sram_expert_ids=sram_expert_ids,
    )
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    # SRAM slots are independent of DRAM list — kernel filters DRAM via bit-15 in gate indices.
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(layer.routed_up_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(layer.routed_down_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    assert len(layer.sram_gate_proj) == len(sram_expert_ids)
    assert len(layer.sram_up_proj) == len(sram_expert_ids)
    assert len(layer.sram_down_proj) == len(sram_expert_ids)
    for ct in (*layer.sram_gate_proj, *layer.sram_up_proj, *layer.sram_down_proj):
        assert isinstance(ct, CompressedTensor)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_moe_layer_weights_empty_sram_expert_ids_4x2(bh_2d_mesh_device):
    """Default (no sram_expert_ids) leaves sram_*_proj empty — SRAM chain skips uniformly."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)

    layer = prepare_moe_layer_weights(submesh, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS)
    assert layer.sram_gate_proj == []
    assert layer.sram_up_proj == []
    assert layer.sram_down_proj == []


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_dense_layer_weights_with_sram_expert_ids_4x2(bh_2d_mesh_device):
    """Dense MLP SRAM chunks populate sram_*_proj; DRAM chunk list stays at 8."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)

    sram_expert_ids = [0, 1, 7]
    # move_to_device=True: dense routed projections go through the TP8 CompressedTensor cache
    # reconstruct callback, which dereferences device.dram_grid_size() and crashes if the cache
    # was told to keep tensors host-side (see expert_dram_memory_config in bspm_expert_cache.py).
    layer = prepare_dense_layer_weights(submesh, state, 0, move_to_device=True, sram_expert_ids=sram_expert_ids)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)
    # Dense routed split: INTERMEDIATE - MOE_INTERMEDIATE chunks. Mirrors test_prepare_routed_expert_weights_dense_4x2.
    expected_chunks = (
        LogicalModelDimensions.INTERMEDIATE_SIZE - LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    ) // LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    assert len(layer.routed_gate_proj) == expected_chunks
    assert len(layer.sram_gate_proj) == len(sram_expert_ids)
    assert len(layer.sram_up_proj) == len(sram_expert_ids)
    assert len(layer.sram_down_proj) == len(sram_expert_ids)
    for ct in (*layer.sram_gate_proj, *layer.sram_up_proj, *layer.sram_down_proj):
        assert isinstance(ct, CompressedTensor)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_moe_layer_weights_compressed_tp8_4x2(bh_2d_mesh_device):
    """compressed_tp8=True returns CompressedTensor routed experts (TP8-sharded) instead of uniform ttnn.Tensor."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)

    layer = prepare_moe_layer_weights(
        submesh,
        state,
        0,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        # move_to_device=True: TP8 path's CompressedTensor cache reconstruct callback dereferences
        # device.dram_grid_size() and crashes when given a None device (i.e. move_to_device=False).
        move_to_device=True,
        compressed_tp8=True,
    )
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS_FOR_TESTS
    for proj_list in (layer.routed_gate_proj, layer.routed_up_proj, layer.routed_down_proj):
        for ct in proj_list:
            assert isinstance(ct, CompressedTensor), f"compressed_tp8 must produce CompressedTensor, got {type(ct)}"


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
# TODO(#43025): Restore standalone final_norm exposure or update this expectation when the interface is resolved.
@pytest.mark.skip(
    reason="[SKIP REASON]: LM-head 4x2 prepare currently returns folded lm_head without standalone final_norm. Issue: #43025"
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
    """Build a synthetic state dict with MTP projection/norm tensors and lm_head."""
    g = torch.Generator().manual_seed(seed + 1000)
    dtype = torch.bfloat16
    H = LogicalModelDimensions.HIDDEN_SIZE
    V = LogicalModelDimensions.VOCAB_SIZE
    return {
        f"model.layers.{mtp_layer_idx}.hnorm.weight": torch.randn(H, generator=g, dtype=dtype),
        f"model.layers.{mtp_layer_idx}.enorm.weight": torch.randn(H, generator=g, dtype=dtype),
        f"model.layers.{mtp_layer_idx}.eh_proj.weight": torch.randn(H, 2 * H, generator=g, dtype=dtype),
        f"model.layers.{mtp_layer_idx}.shared_head.norm.weight": torch.randn(H, generator=g, dtype=dtype),
        "lm_head.weight": torch.randn(V, H, generator=g, dtype=dtype),
    }


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
# TODO(#43025): Root-cause the MTP projection shape mismatch and remove this temporary skip.
@pytest.mark.skip(
    reason="[SKIP REASON]: MTP 4x2 prepare produced eh_projection shape (1792, 7168) instead of expected (14336, 7168). Issue: #43025"
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
# TODO(#43025): Restore standalone shared_head_norm exposure or update this expectation when the interface is resolved.
@pytest.mark.skip(
    reason="[SKIP REASON]: Spec 4x2 prepare currently returns folded lm_head without standalone shared_head_norm. Issue: #43025"
)
def test_prepare_spec_weights_4x2(bh_2d_mesh_device):
    """Prepare spec-stage weights on 4x2 mesh; verify type and shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _mtp_state_dict()
    weights = prepare_spec_weights(state, submesh)
    H = LogicalModelDimensions.HIDDEN_SIZE
    assert isinstance(weights, DeepSeekV3SpecWeights)
    assert weights.shared_head_norm.shape == (1, H)
    assert weights.lm_head is not None


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
# TODO(#43025): Restore standalone final_norm cache artifact or update this expectation when the interface is resolved.
@pytest.mark.skip(
    reason="[SKIP REASON]: LM-head 4x2 TensorCache prepare currently returns folded lm_head without standalone final_norm. Issue: #43025"
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
@requires_hybrid_allocator
def test_prepare_attention_weights_with_cache_dense_4x2(bh_2d_mesh_device, tmp_path):
    """Attention fusion groups (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms) via TensorCache: miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=False)

    attn = prepare_attention_weights(submesh, state, 0, is_moe=False, cache_config=cache_config)
    assert attn.gate_mm is None
    assert attn.q_a_proj.tensor_shape == (3584, 3072)

    _deallocate_attention_weights(attn)

    attn_hit = prepare_attention_weights(submesh, state, 0, is_moe=False, cache_config=cache_config)
    assert attn_hit.gate_mm is None
    assert attn_hit.q_a_proj.tensor_shape == attn.q_a_proj.tensor_shape
    _assert_attention_metadata_matches(attn, attn_hit)

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    # Dense mla_tp==2: kv_b12 + merged_tp4_main (o_proj + norms + q_ab + kv_a). gate_mm is MoE-only.
    assert len(artifact_dirs) >= 2, f"Expected 2 fusion artifacts (kv_b12, merged_tp4_main), found {len(artifact_dirs)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_attention_weights_with_cache_moe_4x2(bh_2d_mesh_device, tmp_path):
    """Attention fusion groups + gate_bias via TensorCache on MoE layer: miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=True, seed=43)

    attn = prepare_attention_weights(submesh, state, 0, is_moe=True, cache_config=cache_config)
    assert attn.gate_mm is not None
    assert attn.gate_bias is not None

    _deallocate_attention_weights(attn)

    attn_hit = prepare_attention_weights(submesh, state, 0, is_moe=True, cache_config=cache_config)
    assert attn_hit.gate_mm is not None
    assert attn_hit.gate_bias.shape == (16, 16)
    _assert_attention_metadata_matches(attn, attn_hit)

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    # MoE mla_tp==2: kv_b12 + merged_tp4_main + merged_tp4_gate (standalone gate_mm) + gate_bias.
    assert (
        len(artifact_dirs) >= 4
    ), f"Expected 3 fusion artifacts (kv_b12, merged_tp4_main, merged_tp4_gate) + gate_bias, found {len(artifact_dirs)}"


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

    shared = prepare_shared_expert_weights(submesh, state, 0, is_moe=False, cache_config=cache_config)
    assert shared.shared_gate_proj.tensor_shape is not None

    ttnn.deallocate(shared.shared_gate_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_up_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_down_proj, force=True)

    shared_hit = prepare_shared_expert_weights(submesh, state, 0, is_moe=False, cache_config=cache_config)
    assert shared_hit.shared_gate_proj.tensor_shape == shared.shared_gate_proj.tensor_shape
    _assert_shared_expert_metadata_matches(shared, shared_hit)

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

    shared = prepare_shared_expert_weights(submesh, state, 0, is_moe=True, cache_config=cache_config)
    assert shared.shared_gate_proj.tensor_shape is not None

    ttnn.deallocate(shared.shared_gate_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_up_proj.fused_tensor, force=True)
    ttnn.deallocate(shared.shared_down_proj, force=True)

    shared_hit = prepare_shared_expert_weights(submesh, state, 0, is_moe=True, cache_config=cache_config)
    assert shared_hit.shared_gate_proj.tensor_shape == shared.shared_gate_proj.tensor_shape
    _assert_shared_expert_metadata_matches(shared, shared_hit)

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

    # move_to_device=True: dense routed CompressedTensor cache reconstruct needs a non-None device.
    routed = prepare_routed_expert_weights(
        submesh, state, 0, is_moe=False, move_to_device=True, cache_config=cache_config
    )
    assert isinstance(routed, DenseRoutedExpertWeights)

    for ct in (*routed.routed_gate_proj, *routed.routed_up_proj, *routed.routed_down_proj):
        for t in ct.get_data_tensors():
            ttnn.deallocate(t, force=True)

    routed_hit = prepare_routed_expert_weights(
        submesh, state, 0, is_moe=False, move_to_device=True, cache_config=cache_config
    )
    assert isinstance(routed_hit, DenseRoutedExpertWeights)

    objects_dir = cache_config.cache.local_root / "objects"
    # Dense routed now routes through prepare_dense_routed_experts_compressed_tp8 →
    # get_or_create_bspm_expert_tp8 → CompressedTensorTarget cache path, which writes
    # tiles.bin (not data.tensorbin) — one per (projection, chunk) entry.
    tiles_bin = list(objects_dir.rglob("tiles.bin"))
    expected_chunks = (
        LogicalModelDimensions.INTERMEDIATE_SIZE - LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    ) // LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    expected_tiles = expected_chunks * 3  # 3 projections (gate/up/down) per chunk
    assert (
        len(tiles_bin) == expected_tiles
    ), f"Expected {expected_tiles} dense TP8 tiles.bin ({expected_chunks} chunks × 3 projections), found {len(tiles_bin)}"


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

    routed = prepare_routed_expert_weights(
        submesh,
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
        submesh,
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
@requires_hybrid_allocator
def test_prepare_dense_layer_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Full dense layer via TensorCache: attention + gate_up + shared_down + routed; miss then hit."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _layer_state_dict(0, is_moe=False)

    # move_to_device=True: dense routed CompressedTensor cache reconstruct needs a non-None device.
    layer = prepare_dense_layer_weights(submesh, state, 0, move_to_device=True, cache_config=cache_config)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)

    _deallocate_layer(layer)

    layer_hit = prepare_dense_layer_weights(submesh, state, 0, move_to_device=True, cache_config=cache_config)
    assert isinstance(layer_hit, DeepSeekV3DenseLayerWeights)
    assert layer_hit.q_a_proj.tensor_shape == layer.q_a_proj.tensor_shape
    _assert_attention_metadata_matches(layer, layer_hit)
    _assert_shared_expert_metadata_matches(layer, layer_hit)

    objects_dir = cache_config.cache.local_root / "objects"
    data_bin = list(objects_dir.rglob("data.tensorbin"))
    tiles_bin = list(objects_dir.rglob("tiles.bin"))
    # Dense mla_tp==2 layer cache splits across two artifact formats:
    #   data.tensorbin (standard TensorTarget/FusionGroupSpec): 2 attn fusion + gate_up + shared_down = 4
    #   tiles.bin (CompressedTensorTarget for dense TP8 routed): 8 chunks × 3 projections = 24
    expected_chunks = (
        LogicalModelDimensions.INTERMEDIATE_SIZE - LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    ) // LogicalModelDimensions.MOE_INTERMEDIATE_SIZE
    expected_tiles = expected_chunks * 3
    assert len(data_bin) >= 4, f"Expected 4 data.tensorbin (2 attn + gate_up + shared_down), found {len(data_bin)}"
    assert (
        len(tiles_bin) == expected_tiles
    ), f"Expected {expected_tiles} tiles.bin (dense TP8 routed), found {len(tiles_bin)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@requires_hybrid_allocator
def test_prepare_moe_layer_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare MoE layer via TensorCache: fusion + gate_bias + gate_up + shared_down + routed experts."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    layer_idx = 3
    state = _layer_state_dict(layer_idx, is_moe=True)

    weights = prepare_moe_layer_weights(
        submesh,
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
        submesh,
        state,
        layer_idx,
        num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
        cache_config=cache_config,
    )
    assert weights_hit.gate_bias.shape == expected_gate_bias
    _assert_attention_metadata_matches(weights, weights_hit)
    _assert_shared_expert_metadata_matches(weights, weights_hit)

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    n_r = NUM_ROUTED_EXPERTS_FOR_TESTS
    # MoE mla_tp==2: kv_b12 + merged_tp4_main + merged_tp4_gate + gate_bias + gate_up + shared_down
    # + n_r * 3 routed = 6 + n_r * 3.
    assert len(artifact_dirs) >= 6 + n_r * 3, (
        f"Expected kv_b12 + merged_tp4_main + merged_tp4_gate + gate_bias + gate_up + shared_down + "
        f"{n_r * 3} routed ({6 + n_r * 3}), found {len(artifact_dirs)}"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
# TODO(#43025): Root-cause the MTP TensorCache projection shape mismatch and remove this temporary skip.
@pytest.mark.skip(
    reason="[SKIP REASON]: MTP 4x2 TensorCache prepare produced eh_projection shape (1792, 7168) instead of expected (14336, 7168). Issue: #43025"
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


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
# TODO(#43025): Restore standalone shared_head_norm cache artifact or update this expectation when the interface is resolved.
@pytest.mark.skip(
    reason="[SKIP REASON]: Spec 4x2 TensorCache prepare currently returns folded lm_head without standalone shared_head_norm. Issue: #43025"
)
def test_prepare_spec_weights_with_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Prepare spec weights via TensorCache on 4x2 mesh: cold miss then warm hit for shared_head_norm."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    cache_config = CacheConfig(cache=TensorCache(tmp_path), context=_test_cache_context())

    state = _mtp_state_dict()
    H = LogicalModelDimensions.HIDDEN_SIZE

    weights = prepare_spec_weights(state, submesh, cache_config=cache_config)
    assert isinstance(weights, DeepSeekV3SpecWeights)
    assert weights.shared_head_norm.shape == (1, H)
    assert weights.lm_head is not None

    expected_norm_shape = weights.shared_head_norm.shape
    expected_lm_shape = weights.lm_head.shape

    ttnn.deallocate(weights.shared_head_norm, force=True)
    ttnn.deallocate(weights.lm_head, force=True)

    weights_hit = prepare_spec_weights(state, submesh, cache_config=cache_config)
    assert weights_hit.shared_head_norm.shape == expected_norm_shape
    assert weights_hit.lm_head.shape == expected_lm_shape

    objects_dir = cache_config.cache.local_root / "objects"
    artifact_dirs = list(objects_dir.rglob("data.tensorbin"))
    assert (
        len(artifact_dirs) >= 2
    ), f"Expected at least 2 cached artifacts (shared_head_norm + lm_head), found {len(artifact_dirs)}"


# =============================================================================
# BSPM / CompressedTensor tests
# =============================================================================

import struct as _struct

_BSPM_STRUCT_FMT = _struct.Struct("<4sIIIIIIIIB3xII")
_BSPM_HEADER_SIZE = 64


def _write_synthetic_bspm(
    path,
    layer_idx: int,
    n_experts: int,
    tile_rows: int,
    tile_cols: int,
    codes_bitsculpt: np.ndarray,
    variant_code: int = 1,
    budget_millibits: int = 3500,
) -> None:
    """Write a minimal valid .bspm binary for integration tests."""
    tiles_per_proj = tile_rows * tile_cols
    header_fields = _BSPM_STRUCT_FMT.pack(
        b"BSPM",
        1,
        layer_idx,
        n_experts,
        3,
        tiles_per_proj,
        tile_rows,
        tile_cols,
        32,
        variant_code,
        budget_millibits,
        budget_millibits,
    )
    header = header_fields + b"\x00" * (_BSPM_HEADER_SIZE - len(header_fields))
    path.write_bytes(header + codes_bitsculpt.astype(np.uint8).tobytes())


def _small_bspm_state_dict(
    layer_idx: int,
    *,
    num_experts: int,
    K: int,
    N: int,
    seed: int = 99,
) -> dict[str, torch.Tensor]:
    """Minimal state dict for BSPM tests using small custom-shaped expert weights.

    HF convention: weight.shape = (out_features, in_features) = (N, K).
    All three projections use the same K/N for simplicity.
    """
    g = torch.Generator().manual_seed(seed)
    state = {}
    for e in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.{proj}.weight"] = torch.randn(
                N, K, generator=g, dtype=torch.bfloat16
            )
    return state


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_routed_experts_bspm_output_types_4x2(bh_2d_mesh_device, tmp_path):
    """prepare_moe_routed_experts_bspm returns CompressedTensor per expert, correct counts,
    correct shapes, DRAM-contiguous; and tiles.bin files are written to TensorCache."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    tile_w = 32
    num_banks = submesh.dram_grid_size().x
    num_experts = 2
    layer_idx = 3
    K, N = 256, 256
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    tiles_h = K // tile_w
    tiles_w_count = N_padded // tile_w
    total_tiles = tiles_h * tiles_w_count

    state = _small_bspm_state_dict(layer_idx, num_experts=num_experts, K=K, N=N)
    # All BFP4 codes (tt-metal code 1) — simplest valid assignment
    codes = np.ones((num_experts, 3, total_tiles), dtype=np.int8)
    bspm_data = {"n_experts": num_experts, "codes": codes}

    cache_config = CacheConfig(
        cache=TensorCache(tmp_path),
        context=_test_cache_context(mesh_shape=(4, 2)),
    )

    routed = prepare_moe_routed_experts_bspm(
        submesh,
        state,
        layer_idx,
        num_experts,
        num_banks,
        bspm_data,
        move_to_device=True,
        cache_config=cache_config,
    )

    assert isinstance(routed, MoERoutedExpertWeights)
    assert len(routed.routed_gate_proj) == num_experts
    assert len(routed.routed_up_proj) == num_experts
    assert len(routed.routed_down_proj) == num_experts

    for e in range(num_experts):
        for proj_name, proj_list in [
            ("routed_gate_proj", routed.routed_gate_proj),
            ("routed_up_proj", routed.routed_up_proj),
            ("routed_down_proj", routed.routed_down_proj),
        ]:
            ct = proj_list[e]
            assert isinstance(ct, CompressedTensor), f"{proj_name}[{e}] is not CompressedTensor"
            assert ct.shape == (K, N_padded), f"{proj_name}[{e}].shape {ct.shape} != ({K}, {N_padded})"

    routed.validate_contiguous_dram()

    # tiles.bin written: one per expert per projection = num_experts * 3
    tiles_bin_files = list((cache_config.cache.local_root / "objects").rglob("tiles.bin"))
    assert (
        len(tiles_bin_files) == num_experts * 3
    ), f"Expected {num_experts * 3} tiles.bin files, found {len(tiles_bin_files)}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_bspm_missing_raises_4x2(bh_2d_mesh_device, tmp_path):
    """bspm_dir set + .bspm file missing raises FileNotFoundError — silent BFP4 fallback was removed
    so a typo'd --bspm-dir argument can't quietly degrade to a different (BFP4-only) model.

    Callers that *do* want uniform BFP4 must pass bspm_dir=None (or omit --bspm-dir) explicitly.
    """
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)

    # tmp_path is empty — no precision_map_B_3.5.bspm file exists for layer 0
    with pytest.raises(FileNotFoundError, match="BSPM file required"):
        prepare_routed_expert_weights(
            submesh,
            state,
            0,
            is_moe=True,
            num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
            move_to_device=True,
            bspm_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_routed_experts_bspm_tile_assignment_4x2(bh_2d_mesh_device, tmp_path):
    """Tile format distribution is preserved end-to-end: counts of BFP4/BFP2/zero codes in the
    returned CompressedTensor._assignment_flat match the input assignment_logical for each expert.

    The DRAM shuffle is a permutation so tile counts are invariant.  This test would catch a
    silent tile-order mismatch from the reshape fix (Bug 24).
    """
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    tile_w = 32
    num_banks = submesh.dram_grid_size().x
    num_experts = 2
    layer_idx = 3
    K, N = 256, 256
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    tiles_h = K // tile_w
    tiles_w_count = N_padded // tile_w
    total_tiles = tiles_h * tiles_w_count

    # Mixed assignment for gate_proj: ~60% BFP4 (1), ~25% BFP2 (2), ~15% zero (3)
    rng = np.random.default_rng(42)
    gate_codes = rng.choice([1, 2, 3], size=(num_experts, total_tiles), p=[0.60, 0.25, 0.15]).astype(np.int8)
    # up/down: uniform BFP4 for simplicity
    up_codes = np.ones((num_experts, total_tiles), dtype=np.int8)
    down_codes = np.ones((num_experts, total_tiles), dtype=np.int8)
    codes = np.stack([gate_codes, up_codes, down_codes], axis=1)  # (num_experts, 3, total_tiles)

    state = _small_bspm_state_dict(layer_idx, num_experts=num_experts, K=K, N=N, seed=77)
    bspm_data = {"n_experts": num_experts, "codes": codes}

    cache_config = CacheConfig(
        cache=TensorCache(tmp_path),
        context=_test_cache_context(mesh_shape=(4, 2)),
    )

    routed = prepare_moe_routed_experts_bspm(
        submesh,
        state,
        layer_idx,
        num_experts,
        num_banks,
        bspm_data,
        move_to_device=True,
        cache_config=cache_config,
    )

    # Shuffle is a permutation — counts per code are invariant.
    for e in range(num_experts):
        ct = routed.routed_gate_proj[e]
        assert isinstance(ct, CompressedTensor)
        actual_flat = ct._assignment_flat  # DRAM-shuffled order, same counts as input
        expected_flat = gate_codes[e]  # logical order; counts are the same after shuffle

        for code, name in [(1, "BFP4"), (2, "BFP2"), (3, "zero")]:
            expected_count = int(np.sum(expected_flat == code))
            actual_count = int(np.sum(actual_flat == code))
            assert (
                actual_count == expected_count
            ), f"expert {e} gate_proj: {name} count mismatch — expected {expected_count}, got {actual_count}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_routed_experts_bspm_footprint_4x2(bh_2d_mesh_device, tmp_path):
    """Compact tiles.bin disk footprint is smaller than uniform BFP4 when assignment is mixed."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    tile_w = 32
    num_banks = submesh.dram_grid_size().x
    num_experts = 2
    layer_idx = 4
    K, N = 256, 256
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    tiles_h = K // tile_w
    tiles_w_count = N_padded // tile_w
    total_tiles = tiles_h * tiles_w_count

    # Mixed assignment: 50% BFP4, 30% BFP2, 20% zero
    rng = np.random.default_rng(17)
    mixed_codes = rng.choice([1, 2, 3], size=(num_experts, total_tiles), p=[0.50, 0.30, 0.20]).astype(np.int8)
    codes = np.stack([mixed_codes, mixed_codes, mixed_codes], axis=1)

    state = _small_bspm_state_dict(layer_idx, num_experts=num_experts, K=K, N=N, seed=55)
    bspm_data = {"n_experts": num_experts, "codes": codes}

    cache_config = CacheConfig(
        cache=TensorCache(tmp_path),
        context=_test_cache_context(mesh_shape=(4, 2)),
    )
    prepare_moe_routed_experts_bspm(
        submesh,
        state,
        layer_idx,
        num_experts,
        num_banks,
        bspm_data,
        move_to_device=True,
        cache_config=cache_config,
    )

    # Each tiles.bin should be smaller than uniform BFP4 baseline
    bfp4_baseline = bfp4_tile_byte_count(tiles_h, tiles_w_count)
    for tiles_bin in (cache_config.cache.local_root / "objects").rglob("tiles.bin"):
        compact_bytes = tiles_bin.stat().st_size
        assert (
            compact_bytes < bfp4_baseline
        ), f"{tiles_bin}: compact={compact_bytes} B >= BFP4 baseline={bfp4_baseline} B"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_layer_bspm_cache_roundtrip_4x2(bh_2d_mesh_device, tmp_path):
    """TensorCache roundtrip for BSPM experts: cold miss writes tiles.bin; warm hit loads from disk.

    Asserts:
    - After cold miss: tiles.bin + assignment.npy written for each expert×proj.
    - After warm hit: no new files written (same count).
    - Both calls return CompressedTensor objects with identical tile-code distributions.
    """
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    tile_w = 32
    num_banks = submesh.dram_grid_size().x
    num_experts = 2
    layer_idx = 5
    K, N = 256, 256
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    tiles_h = K // tile_w
    tiles_w_count = N_padded // tile_w
    total_tiles = tiles_h * tiles_w_count

    rng = np.random.default_rng(99)
    codes_flat = rng.choice([1, 2, 3], size=(num_experts, total_tiles), p=[0.60, 0.25, 0.15]).astype(np.int8)
    codes = np.stack([codes_flat, codes_flat, codes_flat], axis=1)
    state = _small_bspm_state_dict(layer_idx, num_experts=num_experts, K=K, N=N, seed=11)
    bspm_data = {"n_experts": num_experts, "codes": codes}

    cache_config = CacheConfig(
        cache=TensorCache(tmp_path),
        context=_test_cache_context(mesh_shape=(4, 2)),
    )

    # Cold miss
    routed_miss = prepare_moe_routed_experts_bspm(
        submesh,
        state,
        layer_idx,
        num_experts,
        num_banks,
        bspm_data,
        move_to_device=True,
        cache_config=cache_config,
    )
    tiles_after_miss = list((tmp_path / "objects").rglob("tiles.bin"))
    assert (
        len(tiles_after_miss) == num_experts * 3
    ), f"Expected {num_experts * 3} tiles.bin files after cold miss, found {len(tiles_after_miss)}"

    # Warm hit — same cache, same bspm_data
    routed_hit = prepare_moe_routed_experts_bspm(
        submesh,
        state,
        layer_idx,
        num_experts,
        num_banks,
        bspm_data,
        move_to_device=True,
        cache_config=cache_config,
    )
    tiles_after_hit = list((tmp_path / "objects").rglob("tiles.bin"))
    assert len(tiles_after_hit) == num_experts * 3, "Warm hit must not write new tiles.bin files"

    # Code distributions must match between miss and hit
    for e in range(num_experts):
        ct_miss = routed_miss.routed_gate_proj[e]
        ct_hit = routed_hit.routed_gate_proj[e]
        assert isinstance(ct_miss, CompressedTensor)
        assert isinstance(ct_hit, CompressedTensor)
        for code, name in [(1, "BFP4"), (2, "BFP2"), (3, "zero")]:
            count_miss = int(np.sum(ct_miss._assignment_flat == code))
            count_hit = int(np.sum(ct_hit._assignment_flat == code))
            assert count_miss == count_hit, f"expert {e}: {name} count miss={count_miss} vs hit={count_hit}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_via_bspm_dir_4x2(bh_2d_mesh_device, tmp_path):
    """End-to-end integration: synthetic .bspm file on disk → load_bspm_for_layer →
    prepare_routed_expert_weights → CompressedTensor with correct remapped code distribution.

    This exercises the real bspm_loader.py binary parsing and BitSculpt→tt-metal code
    remapping path that is bypassed when bspm_data is injected directly in other tests.
    """
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    tile_w = 32
    num_banks = submesh.dram_grid_size().x
    num_experts = 2
    layer_idx = 6
    K, N = 256, 256
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    tiles_h = K // tile_w  # K/32 — rows in tt-metal convention
    tiles_w_count = N_padded // tile_w  # N/32 — cols in tt-metal convention

    # BSPM stores codes in (N/32, K/32) order (= tiles_w_count × tiles_h).
    # We use BitSculpt codes: 1=bfp2 (→ tt-metal 2) and 2=bfp4 (→ tt-metal 1).
    # Split 50/50 so the remapping is verifiable by count.
    rng = np.random.default_rng(42)
    tiles_per_proj = tiles_w_count * tiles_h  # storage order matches BSPM file
    bs_codes_flat = rng.choice([1, 2], size=(num_experts, 3, tiles_per_proj), p=[0.5, 0.5]).astype(np.uint8)

    # Write the synthetic .bspm into the expected directory layout.
    bspm_dir = tmp_path / "bspm_results"
    bspm_file = bspm_dir / f"layer_{layer_idx}" / "precision_eval" / "precision_map_B_3.5.bspm"
    bspm_file.parent.mkdir(parents=True)
    _write_synthetic_bspm(
        bspm_file,
        layer_idx=layer_idx,
        n_experts=num_experts,
        tile_rows=tiles_w_count,  # BSPM tile_rows = N/32
        tile_cols=tiles_h,  # BSPM tile_cols = K/32
        codes_bitsculpt=bs_codes_flat,
    )

    state = _small_bspm_state_dict(layer_idx, num_experts=num_experts, K=K, N=N)

    routed = prepare_routed_expert_weights(
        submesh,
        state,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_experts,
        move_to_device=True,
        bspm_dir=bspm_dir,
    )

    assert isinstance(routed, MoERoutedExpertWeights)

    # Verify BitSculpt→tt-metal remapping is correct end-to-end:
    # BS 1 (bfp2) → ttnn 2;  BS 2 (bfp4) → ttnn 1.
    for e in range(num_experts):
        ct = routed.routed_gate_proj[e]
        assert isinstance(ct, CompressedTensor), f"expert {e} gate_proj should be CompressedTensor"

        bs_flat = bs_codes_flat[e, 0]  # BitSculpt codes in (N/32, K/32) flat order
        expected_ttnn = (3 - bs_flat.astype(np.int8)).astype(np.int8)  # BS→ttnn: 3-code

        # After reshape+transpose+shuffle, counts must be preserved.
        actual_ttnn_2 = int(np.sum(ct._assignment_flat == 2))  # bfp2 in tt-metal
        actual_ttnn_1 = int(np.sum(ct._assignment_flat == 1))  # bfp4 in tt-metal
        expected_ttnn_2 = int(np.sum(expected_ttnn == 2))
        expected_ttnn_1 = int(np.sum(expected_ttnn == 1))

        assert (
            actual_ttnn_2 == expected_ttnn_2
        ), f"expert {e}: tt-metal bfp2 count {actual_ttnn_2} != expected {expected_ttnn_2}"
        assert (
            actual_ttnn_1 == expected_ttnn_1
        ), f"expert {e}: tt-metal bfp4 count {actual_ttnn_1} != expected {expected_ttnn_1}"
