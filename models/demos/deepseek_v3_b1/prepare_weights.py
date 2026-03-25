# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (full logical shapes for the target
mesh), applies key mapping, transpose, and kv_b split, then passes to
BlitzDecodeWeights which fuses and shards onto the mesh.

When a ``CacheConfig`` is provided, each fusion group / standalone tensor /
routed-expert list is routed through ``TensorCache`` for content-addressed
on-disk caching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    BlitzDecodeWeights,
    OverlappedTensor,
)
from models.demos.deepseek_v3_b1.tensor_cache import _TRANSFORM_VERSION, CacheConfig, Fingerprint

# MoE sender core: hardcoded grid (13, 10) so cache layout is consistent across slow/fast dispatch.
# Sender core = (grid.x - 1, grid.y - 1) = (12, 9); must match test_moe_mlp create_runtime_tensors.
MOE_SENDER_GRID_SIZE = (13, 10)
_GATE_BIAS_TILE = ttnn.Tile([16, 16])


def _make_fp(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
    group_name: str,
    layer_idx: int,
    spec_fingerprints: tuple[str, ...],
) -> Fingerprint:
    return Fingerprint(
        schema_version=1,
        hf_model_id=cache_config.hf_model_id,
        hf_revision=cache_config.hf_revision,
        transform_version=_TRANSFORM_VERSION,
        mesh_shape=mesh_shape,
        group_name=group_name,
        layer_idx=layer_idx,
        spec_fingerprints=spec_fingerprints,
    )


def _qab_kva_spec_fps() -> tuple[str, ...]:
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    return (cfg.q_a_shard_spec.fingerprint(), cfg.q_b_shard_spec.fingerprint(), cfg.kv_a_shard_spec.fingerprint())


def _o_proj_norms_spec_fps() -> tuple[str, ...]:
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
    return (
        cfg.o_proj.fingerprint(),
        cfg.gate_mm.fingerprint(),
        cfg.attn_norm.fingerprint(),
        cfg.q_norm.fingerprint(),
        cfg.kv_norm.fingerprint(),
        cfg.ffn_norm.fingerprint(),
    )


def _kv_b12_spec_fps() -> tuple[str, ...]:
    """Fingerprint from the config-level core range sets + known shapes/dtype.

    KVB12 specs are constructed inline (shape depends on mla_tp) but the
    core grids and sharding strategy are fixed.  Combined with mesh_shape
    in the Fingerprint, this uniquely identifies the layout.
    """
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec

    kv_b1_fp = OverlappedShardSpec(
        core_range_set=cfg.kv_b1_core_range_set,
        raw_tensor_shape=cfg.kv_b1_proj_shape,
        dtype=ttnn.DataType.BFLOAT8_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    kv_b2_fp = OverlappedShardSpec(
        core_range_set=cfg.kv_b2_core_range_set,
        raw_tensor_shape=cfg.kv_b2_proj_shape,
        dtype=ttnn.DataType.BFLOAT8_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    return (kv_b1_fp, kv_b2_fp)


def _gate_up_spec_fps() -> tuple[str, ...]:
    """Fingerprint from the config-level shapes and core grids."""
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedShardSpec

    gate_fp = OverlappedShardSpec(
        core_range_set=cfg.gate_core_range_set,
        raw_tensor_shape=cfg.stacked_shape,
        dtype=ttnn.DataType.BFLOAT4_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    up_fp = OverlappedShardSpec(
        core_range_set=cfg.up_core_range_set,
        raw_tensor_shape=cfg.stacked_shape,
        dtype=ttnn.DataType.BFLOAT4_B,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ).fingerprint()
    return (gate_fp, up_fp)


# Cache artifact types returned by layer_fingerprints
CACHE_TYPE_OVERLAPPED = "overlapped"
CACHE_TYPE_TENSOR = "tensor"
CACHE_TYPE_ROUTED_EXPERTS = "routed_experts"


def layer_fingerprints(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
    layer_idx: int,
    is_moe: bool,
) -> dict[str, tuple[Fingerprint, str]]:
    """Build all fingerprints for a single decoder layer.

    Returns ``{group_name: (fingerprint, cache_type)}`` where *cache_type* is
    one of ``CACHE_TYPE_OVERLAPPED``, ``CACHE_TYPE_TENSOR``, or
    ``CACHE_TYPE_ROUTED_EXPERTS``.
    """
    fp = lambda name, specs: _make_fp(cache_config, mesh_shape, name, layer_idx, specs)
    result: dict[str, tuple[Fingerprint, str]] = {
        "q_ab_kv_a": (fp("q_ab_kv_a", _qab_kva_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "o_proj_gate_mm_norms": (fp("o_proj_gate_mm_norms", _o_proj_norms_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "kv_b12": (fp("kv_b12", _kv_b12_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "gate_up": (fp("gate_up", _gate_up_spec_fps()), CACHE_TYPE_OVERLAPPED),
        "shared_down_proj": (fp("shared_down_proj", ()), CACHE_TYPE_TENSOR),
    }
    if is_moe:
        result["gate_bias"] = (fp("gate_bias", ()), CACHE_TYPE_TENSOR)
        result["routed_experts"] = (fp("routed_experts", ()), CACHE_TYPE_ROUTED_EXPERTS)
    else:
        result["routed_gate_proj"] = (fp("routed_gate_proj", ()), CACHE_TYPE_TENSOR)
        result["routed_up_proj"] = (fp("routed_up_proj", ()), CACHE_TYPE_TENSOR)
        result["routed_down_proj"] = (fp("routed_down_proj", ()), CACHE_TYPE_TENSOR)
    return result


def embedding_fingerprint(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
) -> Fingerprint:
    """Build the fingerprint for the embedding tensor."""
    return _make_fp(cache_config, mesh_shape, "embedding", 0, ())


def lm_head_fingerprints(
    cache_config: CacheConfig,
    mesh_shape: tuple[int, int],
) -> dict[str, tuple[Fingerprint, str]]:
    """Build fingerprints for LM head + final norm."""
    fp = lambda name: _make_fp(cache_config, mesh_shape, name, 0, ())
    return {
        "lm_head": (fp("lm_head"), CACHE_TYPE_TENSOR),
        "final_norm": (fp("final_norm"), CACHE_TYPE_TENSOR),
    }


@dataclass
class AttentionWeights:
    """Attention fusion groups: q_ab_kv_a + kv_b12 + o_proj_gate_mm_norms."""

    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor
    o_proj: OverlappedTensor
    gate_mm: OverlappedTensor | None  # None for dense layers
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor
    gate_bias: ttnn.Tensor | None  # e_score_correction_bias for MoE only


@dataclass
class SharedExpertWeights:
    """Shared expert gate_up fusion group + standalone shared_down_proj."""

    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor


@dataclass
class DenseRoutedExpertWeights:
    """Routed expert weights for dense layers (single tensor per proj)."""

    routed_gate_proj: ttnn.Tensor
    routed_up_proj: ttnn.Tensor
    routed_down_proj: ttnn.Tensor


# Must match MoeRoutedExpertOp.setup_dram_matmul(..., num_subblocks_k=4) for stride checks.
_MOE_DRAM_MATMUL_NUM_SUBBLOCKS_K = 4


def _moe_routed_expert_stride_bytes(weights_tensor: ttnn.Tensor) -> int:
    """Packed DRAM size of one routed expert tensor (bytes), per DRAMStreamingMatmul indexing."""
    shard_spec = weights_tensor.memory_config().shard_spec
    if shard_spec is None:
        raise ValueError("MoE routed expert weights must be sharded (WIDTH_SHARDED DRAM).")
    weights_shard_shape = shard_spec.shape
    K = weights_shard_shape[0]
    tile = weights_tensor.tile
    th, tw = tile.tile_shape[0], tile.tile_shape[1]
    per_core_n = weights_shard_shape[1] // tw
    Kt = K // th
    if Kt % _MOE_DRAM_MATMUL_NUM_SUBBLOCKS_K != 0:
        raise AssertionError(
            f"Kt ({Kt}) must be divisible by num_subblocks_k ({_MOE_DRAM_MATMUL_NUM_SUBBLOCKS_K}) "
            "for MoE DRAM matmul stride check."
        )
    weights_tile_size = tile.get_tile_size(weights_tensor.dtype)
    return Kt * per_core_n * weights_tile_size


def _assert_moe_routed_expert_list_contiguous(tensors: list[ttnn.Tensor], name: str) -> None:
    """Experts must be contiguous in DRAM; see MoERoutedExpertWeights.validate_contiguous_dram."""
    if len(tensors) < 2:
        return
    if not ttnn.is_tensor_storage_on_device(tensors[0]):
        return
    stride = _moe_routed_expert_stride_bytes(tensors[0])
    base = tensors[0].buffer_address()
    for i, t in enumerate(tensors):
        expected = base + i * stride
        actual = t.buffer_address()
        if actual != expected:
            raise AssertionError(
                f"{name}[{i}] DRAM layout not contiguous for DRAMStreamingMatmul: "
                f"expected buffer_address {expected}, got {actual} (stride {stride} bytes per expert). "
                "Allocate all experts of one projection in one batch before the next projection."
            )


@dataclass
class MoERoutedExpertWeights:
    """Routed expert weights for MoE layers (list of tensors, one per expert).

    When on device, each of ``routed_gate_proj``, ``routed_up_proj``, and ``routed_down_proj`` must
    be allocated contiguously in DRAM (see :meth:`validate_contiguous_dram`).
    """

    routed_gate_proj: list[ttnn.Tensor]
    routed_up_proj: list[ttnn.Tensor]
    routed_down_proj: list[ttnn.Tensor]

    def validate_contiguous_dram(self) -> None:
        """Assert experts are contiguous in DRAM for each projection (DRAMStreamingMatmul base+stride)."""
        _assert_moe_routed_expert_list_contiguous(self.routed_gate_proj, "routed_gate_proj")
        _assert_moe_routed_expert_list_contiguous(self.routed_up_proj, "routed_up_proj")
        _assert_moe_routed_expert_list_contiguous(self.routed_down_proj, "routed_down_proj")


@dataclass
class DeepSeekV3DenseLayerWeights:
    """Weights for a dense layer (0..first_k_dense_replace-1).

    Has the 3 attention fusion groups and o_proj + norms (no gate_mm).
    """

    # From get_tt_q_ab_proj_and_kv_a_proj_weights
    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor

    # From get_tt_o_proj_and_gate_mm_weights (no gate_mm for dense)
    o_proj: OverlappedTensor
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor

    # From get_tt_kv_b12_proj_weights
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor

    # From get_tt_mlp_shared_expert_weights
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor

    # From get_tt_mlp_routed_expert_weights (1 DRAM expert per device)
    routed_gate_proj: ttnn.Tensor
    routed_up_proj: ttnn.Tensor
    routed_down_proj: ttnn.Tensor


@dataclass
class DeepSeekV3MoELayerWeights:
    """Weights for an MoE layer (first_k_dense_replace..num_layers-1).

    Extends dense with gate_mm and shared expert projections.
    """

    # From get_tt_q_ab_proj_and_kv_a_proj_weights
    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor

    # From get_tt_o_proj_and_gate_mm_weights (includes gate_mm)
    o_proj: OverlappedTensor
    gate_mm: OverlappedTensor
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor

    # MoE gate e_score_correction_bias (standalone)
    gate_bias: ttnn.Tensor

    # From get_tt_kv_b12_proj_weights
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor

    # From get_tt_moe_shared_expert_weights (replaces get_tt_gate_up_proj_weights)
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor

    # From get_tt_moe_routed_expert_weights (256 DRAM experts)
    routed_gate_proj: list[ttnn.Tensor]
    routed_up_proj: list[ttnn.Tensor]
    routed_down_proj: list[ttnn.Tensor]


@dataclass
class DeepSeekV3EmbeddingLayerWeights:
    """Weights for the embedding layer."""

    embedding: ttnn.Tensor


@dataclass
class DeepSeekV3LMHeadWeights:
    """Weights for the LM head and final RMSNorm."""

    lm_head: ttnn.Tensor
    final_norm: ttnn.Tensor  # model.norm.weight, (1, 7168)


# Constants for kv_b_proj split (HF stores one matrix; we split into kv_b1 and kv_b2).
_NUM_HEADS = 64
# MoE routed experts (DeepSeek V3 config: n_routed_experts=256).
NUM_ROUTED_EXPERTS = 256
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256
_Q_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192


def deinterleave_q_b_proj(q_b_proj: torch.Tensor, num_heads: int | None = None) -> torch.Tensor:
    """Convert q_b_proj.weight from HF interleaved to [ALL_NOPE | ALL_ROPE] layout.

    HF stores q_b_proj with out_features = num_heads * q_head_dim, where each head's
    nope and rope dims are contiguous: [h0_nope|h0_rope|h1_nope|h1_rope|...].
    After .T the columns follow this interleaved order.

    The b1 pipeline expects columns grouped as [ALL_NOPE | ALL_ROPE]:
    [h0_nope|h1_nope|...|hN_nope|h0_rope|h1_rope|...|hN_rope].

    Args:
        q_b_transposed: q_b_proj.weight.T with shape (K, num_heads * q_head_dim).
        num_heads: Number of attention heads.  If None, inferred from the width.

    Returns:
        Tensor of the same shape with columns reordered to [ALL_NOPE | ALL_ROPE].
    """
    q_b_transposed = q_b_proj.T
    K, N = q_b_transposed.shape
    if num_heads is None:
        num_heads = N // _Q_HEAD_DIM
    heads = q_b_transposed.reshape(K, num_heads, _Q_HEAD_DIM)
    nope = heads[:, :, :_QK_NOPE_HEAD_DIM].reshape(K, -1)
    rope = heads[:, :, _QK_NOPE_HEAD_DIM:].reshape(K, -1)
    return torch.cat([nope, rope], dim=1).contiguous()


def _key(layer_idx: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_idx}."""
    return f"model.layers.{layer_idx}.{suffix}"


def create_gate_bias_tensor(raw_tensor: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Build gate_bias (e_score_correction_bias) as HEIGHT_SHARDED on sender core, replicated across mesh.

    raw_tensor: shape (256,) from state dict (model.layers.{i}.mlp.gate.e_score_correction_bias).
    Returns ttnn.Tensor with layout expected by MoE op: (16, 16) on sender core, tile 16x16.
    Sender core uses MOE_SENDER_GRID_SIZE so cache layout is consistent across slow/fast dispatch.
    When move_to_device is False (default), tensor is not placed (device=None) for cache generation.
    When move_to_device is True, tensor is placed on device so is_sharded() is true for runtime use.
    """
    sender_core = ttnn.CoreCoord(MOE_SENDER_GRID_SIZE[0] - 1, MOE_SENDER_GRID_SIZE[1] - 1)
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    gate_bias_reshaped = raw_tensor.reshape(16, 16).T.contiguous().to(torch.bfloat16)
    gate_bias_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_core_grid, (16, 16), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        gate_bias_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device if move_to_device else None,
        memory_config=gate_bias_mem_config,
        tile=_GATE_BIAS_TILE,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split HF kv_b_proj (out_features, in_features) into kv_b1 and kv_b2.

    Expects full logical shape (32768, 512) for 4x2 mesh.
    out_features = num_heads * (qk_nope_head_dim + v_head_dim) = num_heads * 256.
    Reshape to (num_heads, 256, 512); first 128 dims are k (b1), last 128 are v (b2).
    Only kv_b2 is transposed for blitz.
    """
    out_features, kv_lora_rank = kv_b_proj.shape
    assert kv_lora_rank == _KV_LORA_RANK
    num_heads = out_features // _KV_B_PROJ_HEAD_DIM
    w = kv_b_proj.reshape(num_heads, _KV_B_PROJ_HEAD_DIM, _KV_LORA_RANK).contiguous()
    kv_b1 = w[:, :_QK_NOPE_HEAD_DIM, :].reshape(-1, _KV_LORA_RANK)
    kv_b2 = w[:, _QK_NOPE_HEAD_DIM:, :].reshape(-1, _KV_LORA_RANK).T.contiguous()
    return kv_b1, kv_b2


# Per-TP attention tensor dimensions (match BlitzDecodeWeights configs for single device)
_MLA_TP1_Q_B_WIDTH = 12288
_MLA_TP1_O_PROJ_HEIGHT = 8192
_MLA_TP1_KV_B1_HEIGHT = 8192
_MLA_TP1_KV_B2_WIDTH = 8192

# Per-TP shared expert dimensions (gate/up (7168, 256), down (256, 7168) for moe_tp=1)
_MOE_TP1_SHARED_GATE_UP_N = 256
_MOE_TP1_SHARED_DOWN_K = 256


def _slice_attention_weights_for_mla_tp(
    q_b: torch.Tensor,
    o_proj: torch.Tensor,
    kv_b1: torch.Tensor,
    kv_b2: torch.Tensor,
    mla_tp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """When state dict has full (2-TP) logical shapes and mla_tp==1, slice to single-TP.

    Single-device tests use mla_tp=1; the reference state dict uses full logical
    shapes (24576 q_b, 16384 o_proj, etc.). Slice to per-TP so BlitzDecodeWeights
    receives the shapes it expects.
    """
    if mla_tp > 1:
        return q_b, o_proj, kv_b1, kv_b2
    # Full logical: q_b (1536, 24576), o_proj (16384, 7168), kv_b1 (16384, 512), kv_b2 (512, 16384)
    if q_b.shape[1] == _MLA_TP1_Q_B_WIDTH * 2:
        q_b = q_b[:, :_MLA_TP1_Q_B_WIDTH].contiguous()
    if o_proj.shape[0] == _MLA_TP1_O_PROJ_HEIGHT * 2:
        o_proj = o_proj[:_MLA_TP1_O_PROJ_HEIGHT, :].contiguous()
    if kv_b1.shape[0] == _MLA_TP1_KV_B1_HEIGHT * 2:
        kv_b1 = kv_b1[:_MLA_TP1_KV_B1_HEIGHT, :].contiguous()
    if kv_b2.shape[1] == _MLA_TP1_KV_B2_WIDTH * 2:
        kv_b2 = kv_b2[:, :_MLA_TP1_KV_B2_WIDTH].contiguous()
    return q_b, o_proj, kv_b1, kv_b2


def _slice_shared_expert_weights_for_moe_tp(
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    moe_tp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """When state dict has full (8-TP) logical shapes and moe_tp==1, slice to single-TP.

    Single-device tests use moe_tp=1; the reference state dict uses full logical
    shapes (gate/up width 2048, down height 2048). Slice to per-TP so BlitzDecodeWeights
    receives (7168, 256) and (256, 7168).
    """
    if moe_tp > 1:
        return shared_gate, shared_up, shared_down
    full_n = _MOE_TP1_SHARED_GATE_UP_N * 8  # 2048
    if shared_gate.shape[1] == full_n:
        shared_gate = shared_gate[:, :_MOE_TP1_SHARED_GATE_UP_N].contiguous()
    if shared_up.shape[1] == full_n:
        shared_up = shared_up[:, :_MOE_TP1_SHARED_GATE_UP_N].contiguous()
    if shared_down.shape[0] == full_n:
        shared_down = shared_down[:_MOE_TP1_SHARED_DOWN_K, :].contiguous()
    return shared_gate, shared_up, shared_down


def get_layer_raw_tensors(
    state_dict: dict[str, torch.Tensor], layer_idx: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract and transform raw tensors for one layer from the state dict.

    Expects full logical HF shapes. We transpose HF
    (out_features, in_features) to (K, N); norms unsqueeze(0) to
    (1, W); kv_b_proj is split into kv_b1 and kv_b2 (see split_kv_b_proj).

    Transformation (HF full logical -> transform -> passed to BlitzDecodeWeights):

        Weight        | HF key (under model.layers.{i}.)     | HF shape      | Transform   | To blitz
        --------------|-------------------------------------|---------------|-------------|------------------
        q_b_proj      | self_attn.q_b_proj.weight            | (24576, 1536) | .T + deinterleave | (1536, 24576) [ALL_NOPE|ALL_ROPE]
        o_proj        | self_attn.o_proj.weight              | (7168, 16384) | .T          | (16384, 7168)
        kv_b_proj     | self_attn.kv_b_proj.weight           | (32768, 512)  | split       | kv_b1, kv_b2
        q_a_proj      | self_attn.q_a_proj.weight            | (1536, 7168)  | .T          | (7168, 1536)
        kv_a_proj     | self_attn.kv_a_proj_with_mqa.weight  | (576, 7168)   | .T          | (7168, 576)
        norms         | input_layernorm, q_a_layernorm, etc. | (7168,), …    | unsqueeze(0)| (1, 7168), …

    MoE-only (gate_mm, shared_gate_proj, shared_up_proj) are read in
    prepare_moe_layer_weights.

    Returns:
        (q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm).
    """
    q_a = state_dict[_key(layer_idx, "self_attn.q_a_proj.weight")].T.contiguous()
    q_b = deinterleave_q_b_proj(state_dict[_key(layer_idx, "self_attn.q_b_proj.weight")])
    kv_a = state_dict[_key(layer_idx, "self_attn.kv_a_proj_with_mqa.weight")].T.contiguous()
    kv_b1, kv_b2 = split_kv_b_proj(state_dict[_key(layer_idx, "self_attn.kv_b_proj.weight")])
    o_proj = state_dict[_key(layer_idx, "self_attn.o_proj.weight")].T.contiguous()

    attn_norm = state_dict[_key(layer_idx, "input_layernorm.weight")].unsqueeze(0)
    q_norm = state_dict[_key(layer_idx, "self_attn.q_a_layernorm.weight")].unsqueeze(0)
    kv_norm = state_dict[_key(layer_idx, "self_attn.kv_a_layernorm.weight")].unsqueeze(0)
    ffn_norm = state_dict[_key(layer_idx, "post_attention_layernorm.weight")].unsqueeze(0)

    return q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm


# Gate routing constants (bias/indices layout on sender core)
_GATE_BIAS_INDICES_SHAPE = (16, 16)
_GATE_NUM_INDICES = 256


def create_gate_indices_tensor(
    device: Any,
    sender_core_grid: ttnn.CoreRangeSet,
    *,
    mesh_mapper: Any = None,
) -> ttnn.Tensor:
    """Build constant gate indices 0..255 as HEIGHT_SHARDED on sender core.

    Same layout as gate_bias: (16, 16), HEIGHT_SHARDED, tile 16x16, uint16.
    """
    indices = torch.arange(_GATE_NUM_INDICES, dtype=torch.int32).reshape(
        _GATE_BIAS_INDICES_SHAPE[0], _GATE_BIAS_INDICES_SHAPE[1]
    )
    transposed = torch.transpose(indices, 0, 1).contiguous().to(torch.uint16)
    shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        _GATE_BIAS_INDICES_SHAPE,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}
    return ttnn.from_torch(
        transposed,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=ttnn.Tile([16, 16]),
        **kwargs,
    )


def prepare_attention_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> AttentionWeights:
    """Prepare attention fusion groups for one layer (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms).

    When ``cache_config`` is provided, each fusion group is routed through
    :meth:`TensorCache.get_or_create`.  On a hit the raw tensors are still
    loaded (cheap) but the expensive tilize-pack-fuse step is skipped.
    """
    mesh_shape = (bdw._device.shape[0], bdw._device.shape[1])

    logger.debug("Loading raw tensors from state dict for layer {}", layer_idx)
    t0 = time.perf_counter()
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = get_layer_raw_tensors(
        state_dict, layer_idx
    )
    q_b, o_proj, kv_b1, kv_b2 = _slice_attention_weights_for_mla_tp(q_b, o_proj, kv_b1, kv_b2, bdw.mla_tp)
    logger.debug("  load raw tensors: {:.3f}s", time.perf_counter() - t0)
    logger.debug("Converting attention fusion groups for layer {} (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)", layer_idx)
    t0 = time.perf_counter()

    def _cached_or_fuse(group_name, spec_fps_fn, fuse_fn):
        if cache_config is not None:
            fp = _make_fp(cache_config, mesh_shape, group_name, layer_idx, spec_fps_fn())
            return cache_config.cache.get_or_create(fp, fuse=fuse_fn, device=bdw._device)
        return fuse_fn()

    def _cached_or_create(tensor_name, create_fn):
        if cache_config is not None:
            fp = _make_fp(cache_config, mesh_shape, tensor_name, layer_idx, ())
            return cache_config.cache.get_or_create_tensor(fp, create=create_fn, device=bdw._device)
        return create_fn()

    qab_kva = _cached_or_fuse(
        "q_ab_kv_a",
        _qab_kva_spec_fps,
        lambda: bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a, q_b, kv_a, move_to_device=move_to_device),
    )
    q_a_proj, q_b_proj, kv_a_proj = qab_kva["q_a_proj"], qab_kva["q_b_proj"], qab_kva["kv_a_proj"]

    kv_b12 = _cached_or_fuse(
        "kv_b12",
        _kv_b12_spec_fps,
        lambda: bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2, move_to_device=move_to_device),
    )
    kv_b1_proj, kv_b2_proj = kv_b12["kv_b1_proj"], kv_b12["kv_b2_proj"]
    logger.debug("  convert q_ab_kv_a + kv_b12: {:.3f}s", time.perf_counter() - t0)

    if is_moe:
        gate_mm = state_dict[_key(layer_idx, "mlp.gate.weight")].T.contiguous()
        o_norms = _cached_or_fuse(
            "o_proj_gate_mm_norms",
            _o_proj_norms_spec_fps,
            lambda: bdw.get_tt_o_proj_and_gate_mm_weights(
                o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
            ),
        )
        gate_bias_raw = state_dict[_key(layer_idx, "mlp.gate.e_score_correction_bias")]
        gate_bias_tt = _cached_or_create(
            "gate_bias",
            lambda: create_gate_bias_tensor(gate_bias_raw, bdw._device, move_to_device=move_to_device),
        )
        logger.debug("  convert o_proj_gate_mm_norms (MoE): {:.3f}s", time.perf_counter() - t0)
        return AttentionWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_norms["o_proj"],
            gate_mm=o_norms["gate_mm"],
            attn_norm=o_norms["attn_norm"],
            q_norm=o_norms["q_norm"],
            kv_norm=o_norms["kv_norm"],
            ffn_norm=o_norms["ffn_norm"],
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            gate_bias=gate_bias_tt,
        )
    else:
        gate_mm_dummy = torch.zeros(7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device)
        o_norms = _cached_or_fuse(
            "o_proj_gate_mm_norms",
            _o_proj_norms_spec_fps,
            lambda: bdw.get_tt_o_proj_and_gate_mm_weights(
                o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
            ),
        )
        logger.debug("  convert o_proj_gate_mm_norms (dense): {:.3f}s", time.perf_counter() - t0)
        return AttentionWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_norms["o_proj"],
            gate_mm=None,
            attn_norm=o_norms["attn_norm"],
            q_norm=o_norms["q_norm"],
            kv_norm=o_norms["kv_norm"],
            ffn_norm=o_norms["ffn_norm"],
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            gate_bias=None,
        )


def prepare_shared_expert_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> SharedExpertWeights:
    """Prepare shared expert weights (gate_up fusion group + shared_down_proj) for one layer.

    When ``cache_config`` is provided, the gate_up fusion group is routed
    through :meth:`TensorCache.get_or_create`.  The standalone down_proj
    tensor is not cached (Phase 2 standalone caching).
    """
    mesh_shape = (bdw._device.shape[0], bdw._device.shape[1])
    logger.debug("Converting shared expert weights for layer {} (is_moe={})", layer_idx, is_moe)
    t0 = time.perf_counter()

    if is_moe:
        shared_gate = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
        shared_up = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()
        shared_down = state_dict[_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T.contiguous()
        shared_gate, shared_up, shared_down = _slice_shared_expert_weights_for_moe_tp(
            shared_gate, shared_up, shared_down, bdw.moe_tp
        )
        fuse_gate_up = lambda: bdw.get_tt_gate_up_proj_weights(shared_gate, shared_up, move_to_device=move_to_device)
        create_down = lambda: bdw.get_tt_shared_down_proj_weights(shared_down, move_to_device=move_to_device)
    else:
        shared_n = 2048
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()[:, :shared_n]
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()[:, :shared_n]
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()[:shared_n, :]
        fuse_gate_up = lambda: bdw.get_tt_gate_up_proj_weights(mlp_gate, mlp_up, move_to_device=move_to_device)
        create_down = lambda: bdw.get_tt_shared_down_proj_weights(mlp_down, move_to_device=move_to_device)

    def _cached_or_fuse(group_name, spec_fps_fn, fuse_fn):
        if cache_config is not None:
            fp = _make_fp(cache_config, mesh_shape, group_name, layer_idx, spec_fps_fn())
            return cache_config.cache.get_or_create(fp, fuse=fuse_fn, device=bdw._device)
        return fuse_fn()

    def _cached_or_create(tensor_name, create_fn):
        if cache_config is not None:
            fp = _make_fp(cache_config, mesh_shape, tensor_name, layer_idx, ())
            return cache_config.cache.get_or_create_tensor(fp, create=create_fn, device=bdw._device)
        return create_fn()

    gate_up = _cached_or_fuse("gate_up", _gate_up_spec_fps, fuse_gate_up)
    shared_down_proj = _cached_or_create("shared_down_proj", create_down)

    logger.debug("  shared expert weights done in {:.3f}s", time.perf_counter() - t0)
    return SharedExpertWeights(
        shared_gate_proj=gate_up["gate_proj"],
        shared_up_proj=gate_up["up_proj"],
        shared_down_proj=shared_down_proj,
    )


def prepare_routed_expert_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DenseRoutedExpertWeights | MoERoutedExpertWeights:
    """Prepare routed expert weights for one layer (dense: single MLP; MoE: num_routed_experts experts).

    When ``cache_config`` is provided:
    * **MoE** — routed through :meth:`TensorCache.get_or_create_routed_experts`
      (768 individual ``.tensorbin`` files with a ``_complete`` sentinel).
    * **Dense** — each of the three projections is routed through
      :meth:`TensorCache.get_or_create_tensor`.
    """
    mesh_shape = (bdw._device.shape[0], bdw._device.shape[1])

    if is_moe:

        def _create_moe() -> MoERoutedExpertWeights:
            logger.info(
                "Loading and converting {} routed experts for layer {} (this may be slow)...",
                num_routed_experts,
                layer_idx,
            )
            t0 = time.perf_counter()
            gate_list = []
            up_list = []
            down_list = []
            for e in range(num_routed_experts):
                if e > 0 and e % 64 == 0:
                    logger.debug("  loaded experts 0..{} from state dict", e - 1)
                gate_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")].T.contiguous())
                up_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")].T.contiguous())
                down_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")].T.contiguous())
            load_elapsed = time.perf_counter() - t0
            logger.info("  loaded {} experts from state dict in {:.3f}s", num_routed_experts, load_elapsed)
            logger.debug("Converting routed experts to device format (blitz)...")
            t0 = time.perf_counter()
            gate_stacked = torch.stack(gate_list, dim=0)
            up_stacked = torch.stack(up_list, dim=0)
            down_stacked = torch.stack(down_list, dim=0)
            routed_gate_proj, routed_up_proj, routed_down_proj = bdw.get_tt_moe_routed_expert_weights(
                gate_stacked, up_stacked, down_stacked, move_to_device=move_to_device
            )
            logger.info("  converted routed experts in {:.3f}s", time.perf_counter() - t0)
            routed = MoERoutedExpertWeights(
                routed_gate_proj=routed_gate_proj,
                routed_up_proj=routed_up_proj,
                routed_down_proj=routed_down_proj,
            )
            if move_to_device:
                routed.validate_contiguous_dram()
            return routed

        if cache_config is not None:
            fp = _make_fp(cache_config, mesh_shape, "routed_experts", layer_idx, ())
            return cache_config.cache.get_or_create_routed_experts(
                fp, create=_create_moe, device=bdw._device, num_experts=num_routed_experts
            )
        return _create_moe()
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()

        if cache_config is not None:
            _dense_cache: dict[str, ttnn.Tensor] = {}

            def _ensure_dense_converted():
                if not _dense_cache:
                    g, u, d = bdw.get_tt_mlp_routed_expert_weights(
                        mlp_gate, mlp_up, mlp_down, move_to_device=move_to_device
                    )
                    _dense_cache["gate"] = g
                    _dense_cache["up"] = u
                    _dense_cache["down"] = d

            def _cached_or_create(tensor_name, key):
                fp = _make_fp(cache_config, mesh_shape, tensor_name, layer_idx, ())

                def _create():
                    _ensure_dense_converted()
                    return _dense_cache[key]

                return cache_config.cache.get_or_create_tensor(fp, create=_create, device=bdw._device)

            routed_gate_proj = _cached_or_create("routed_gate_proj", "gate")
            routed_up_proj = _cached_or_create("routed_up_proj", "up")
            routed_down_proj = _cached_or_create("routed_down_proj", "down")
        else:
            routed_gate_proj, routed_up_proj, routed_down_proj = bdw.get_tt_mlp_routed_expert_weights(
                mlp_gate, mlp_up, mlp_down, move_to_device=move_to_device
            )
        return DenseRoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )


def prepare_dense_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3DenseLayerWeights:
    """Prepare fused weights for a single dense decoder layer."""
    logger.info("Preparing dense layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(
        bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    shared = prepare_shared_expert_weights(
        bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    routed = prepare_routed_expert_weights(
        bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    assert isinstance(routed, DenseRoutedExpertWeights)
    return DeepSeekV3DenseLayerWeights(
        q_a_proj=attn.q_a_proj,
        q_b_proj=attn.q_b_proj,
        kv_a_proj=attn.kv_a_proj,
        o_proj=attn.o_proj,
        attn_norm=attn.attn_norm,
        q_norm=attn.q_norm,
        kv_norm=attn.kv_norm,
        ffn_norm=attn.ffn_norm,
        kv_b1_proj=attn.kv_b1_proj,
        kv_b2_proj=attn.kv_b2_proj,
        shared_gate_proj=shared.shared_gate_proj,
        shared_up_proj=shared.shared_up_proj,
        shared_down_proj=shared.shared_down_proj,
        routed_gate_proj=routed.routed_gate_proj,
        routed_up_proj=routed.routed_up_proj,
        routed_down_proj=routed.routed_down_proj,
    )
    logger.info("  dense layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)


def prepare_moe_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3MoELayerWeights:
    """Prepare fused weights for a single MoE decoder layer."""
    logger.info("Preparing MoE layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(
        bdw, state_dict, layer_idx, is_moe=True, move_to_device=move_to_device, cache_config=cache_config
    )
    shared = prepare_shared_expert_weights(
        bdw, state_dict, layer_idx, is_moe=True, move_to_device=move_to_device, cache_config=cache_config
    )
    routed = prepare_routed_expert_weights(
        bdw,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_routed_experts,
        move_to_device=move_to_device,
        cache_config=cache_config,
    )
    assert isinstance(attn.gate_mm, OverlappedTensor)
    assert attn.gate_bias is not None
    assert isinstance(routed, MoERoutedExpertWeights)
    return DeepSeekV3MoELayerWeights(
        q_a_proj=attn.q_a_proj,
        q_b_proj=attn.q_b_proj,
        kv_a_proj=attn.kv_a_proj,
        o_proj=attn.o_proj,
        gate_mm=attn.gate_mm,
        attn_norm=attn.attn_norm,
        q_norm=attn.q_norm,
        kv_norm=attn.kv_norm,
        ffn_norm=attn.ffn_norm,
        gate_bias=attn.gate_bias,
        kv_b1_proj=attn.kv_b1_proj,
        kv_b2_proj=attn.kv_b2_proj,
        shared_gate_proj=shared.shared_gate_proj,
        shared_up_proj=shared.shared_up_proj,
        shared_down_proj=shared.shared_down_proj,
        routed_gate_proj=routed.routed_gate_proj,
        routed_up_proj=routed.routed_up_proj,
        routed_down_proj=routed.routed_down_proj,
    )
    logger.info("  MoE layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)


def _to_tt_embedding(embedding_torch: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Convert a torch embedding tensor to TT (DRAM, ROW_MAJOR, ReplicateTensorToMesh). Shared by prepare and synthetic."""
    return ttnn.from_torch(
        embedding_torch.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device if move_to_device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def prepare_embedding_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    *,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3EmbeddingLayerWeights:
    """Prepare embedding weights from state dict (model.embed_tokens.weight)."""
    logger.info("Preparing embedding weights...")
    w = state_dict["model.embed_tokens.weight"]
    assert w.shape == (129280, 7168), f"Expected embedding shape (129280, 7168), got {w.shape}"

    def create():
        return _to_tt_embedding(w, device, move_to_device=move_to_device)

    if cache_config is not None:
        mesh_shape = (device.shape[0], device.shape[1])
        fp = _make_fp(cache_config, mesh_shape, "embedding", 0, ())
        embedding_tt = cache_config.cache.get_or_create_tensor(fp, create=create, device=device)
    else:
        embedding_tt = create()
    return DeepSeekV3EmbeddingLayerWeights(embedding=embedding_tt)


# LM head: HF keeps full vocab (129280, 7168). Prepare shards vocab (N) across the mesh (TP=mesh size)
# and uses the same per-device L1 WIDTH_SHARDED layout as test_lm_head_sampling (101 matmul cores).

_LM_HEAD_K = 7168
_LM_HEAD_VOCAB_SIZE = 129280
_LM_HEAD_NUM_MATMUL_CORES = 101
_LM_HEAD_MATMUL_CORE_GRID = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
    ]
)
_LM_HEAD_B_TILE = ttnn.Tile([32, 32])
_LM_HEAD_A_TILE = ttnn.Tile([1, 32])
_LM_HEAD_N_PER_CORE = 160
_LM_HEAD_MCAST_CORE = ttnn.CoreCoord(10, 9)
_LM_HEAD_MCAST_CORE_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(_LM_HEAD_MCAST_CORE, _LM_HEAD_MCAST_CORE)])


def _to_tt_lm_head_matrix(
    lm_head_torch: torch.Tensor, device, *, mesh_mapper, move_to_device: bool = False
) -> ttnn.Tensor:
    """Convert (K, N) lm_head torch tensor to TT (WIDTH_SHARDED 101 cores, L1). Shared by prepare and synthetic."""
    lm_head_shard_shape = (_LM_HEAD_K, _LM_HEAD_N_PER_CORE)
    lm_head_shard_spec = ttnn.ShardSpec(
        _LM_HEAD_MATMUL_CORE_GRID,
        lm_head_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    lm_head_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        lm_head_shard_spec,
    )
    return ttnn.from_torch(
        lm_head_torch.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device if move_to_device else None,
        memory_config=lm_head_mem_config,
        mesh_mapper=mesh_mapper,
        tile=_LM_HEAD_B_TILE,
    )


def _to_tt_lm_head_final_norm(norm_torch: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Convert (1, K) final norm torch tensor to TT (HEIGHT_SHARDED on mcast core). Shared by prepare and synthetic."""
    norm_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_LM_HEAD_MCAST_CORE_GRID, (1, _LM_HEAD_K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        norm_torch.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=_LM_HEAD_A_TILE,
        device=device if move_to_device else None,
        memory_config=norm_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def prepare_lm_head_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    *,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3LMHeadWeights:
    """Prepare LM head and final norm weights from state dict.

    device must be the mesh device (e.g. 4x2 submesh). The LM head weight matrix is sharded
    along the vocabulary dimension (TP = mesh size). Per-device layout matches the LM head
    sampling op: WIDTH_SHARDED in L1 across 101 matmul cores with shard shape (7168, N_per_core).
    """
    logger.info("Preparing LM head weights...")
    lm_w = state_dict["lm_head.weight"]
    assert lm_w.shape == (
        _LM_HEAD_VOCAB_SIZE,
        _LM_HEAD_K,
    ), f"Expected lm_head shape ({_LM_HEAD_VOCAB_SIZE}, {_LM_HEAD_K}), got {lm_w.shape}"

    logger.info("Preparing LM head norm...")
    norm_w = state_dict["model.norm.weight"]
    assert norm_w.shape == (7168,), f"Expected final norm shape (7168,), got {norm_w.shape}"

    def create_lm_head():
        return _to_tt_lm_head_matrix(
            lm_w.T, device, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1), move_to_device=move_to_device
        )

    def create_final_norm():
        return _to_tt_lm_head_final_norm(norm_w.unsqueeze(0), device, move_to_device=move_to_device)

    if cache_config is not None:
        mesh_shape = (device.shape[0], device.shape[1])
        fp_lm = _make_fp(cache_config, mesh_shape, "lm_head", 0, ())
        lm_head_tt = cache_config.cache.get_or_create_tensor(fp_lm, create=create_lm_head, device=device)
        fp_norm = _make_fp(cache_config, mesh_shape, "final_norm", 0, ())
        final_norm_tt = cache_config.cache.get_or_create_tensor(fp_norm, create=create_final_norm, device=device)
    else:
        lm_head_tt = create_lm_head()
        final_norm_tt = create_final_norm()

    return DeepSeekV3LMHeadWeights(lm_head=lm_head_tt, final_norm=final_norm_tt)
