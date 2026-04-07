# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (full logical shapes for the target
mesh), applies key mapping, transpose, and kv_b split, then passes to
BlitzDecodeWeights which fuses and shards onto the mesh.

For on-disk persistence use :class:`~models.demos.deepseek_v3_b1.tensor_cache.TensorCache`
with :class:`~models.demos.deepseek_v3_b1.tensor_cache.CacheConfig` and the ``prepare_*`` functions
(see ``demo/weight_provider.py``). Legacy ``layer_NNN/manifest.json`` layouts are not supported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    DOWN_PROJ_SINGLE_DEVICE_SPEC,
    GATE_UP_SPEC,
    KV_B12_SPEC,
    O_PROJ_GATE_MM_NORMS_SPEC,
    Q_AB_KV_A_SPEC,
    BlitzDecodeWeights,
    OverlappedTensor,
)
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.tensor_cache import (
    CacheConfig,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SourceTensorSelection,
    TensorTarget,
)

# Bump when any standalone tensor preprocessing logic changes to invalidate caches.
CURRENT_TRANSFORM_VERSION = 1

# MoE sender core: hardcoded grid (13, 10) so cache layout is consistent across slow/fast dispatch.
# Sender core = (grid.x - 1, grid.y - 1) = (12, 9); must match test_moe_mlp create_runtime_tensors.
MOE_SENDER_GRID_SIZE = (13, 10)
_GATE_BIAS_TILE = ttnn.Tile([16, 16])


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


@dataclass
class DeepSeekV3MTPWeights:
    """Weights for the MTP (Multi-Token Prediction) speculative decode layer.

    HF state dict keys live under ``model.layers.{mtp_layer_idx}.*`` (layer 61 for DeepSeek V3).
    The MTP decoder block (layer 61) is a regular MoE layer loaded separately.
    """

    h_gamma: ttnn.Tensor  # model.layers.61.hnorm.weight
    e_gamma: ttnn.Tensor  # model.layers.61.enorm.weight
    eh_projection: ttnn.Tensor  # model.layers.61.eh_proj.weight


# Constants for kv_b_proj split (HF stores one matrix; we split into kv_b1 and kv_b2).
_NUM_HEADS = 64

# MoE routed experts (DeepSeek V3 config: n_routed_experts=256).
NUM_ROUTED_EXPERTS = D.GATE_NUM_INDICES
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_KV_LORA_RANK = D.KV_B_LORA_RANK
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256
_Q_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192

# MTP layer constants
_MTP_LAYER_IDX = 61
_MTP_NUM_DRAM_BANKS = 8
_MTP_B_TILE = ttnn.Tile([32, 32])

_GATE_BIAS_SENDER_CORE = ttnn.CoreCoord(MOE_SENDER_GRID_SIZE[0] - 1, MOE_SENDER_GRID_SIZE[1] - 1)
_GATE_BIAS_SENDER_CORE_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(_GATE_BIAS_SENDER_CORE, _GATE_BIAS_SENDER_CORE)])

_LM_HEAD_K = D.HIDDEN_SIZE
_LM_HEAD_VOCAB_SIZE = D.VOCAB_SIZE
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

_NORM_MEM_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(_LM_HEAD_MCAST_CORE_GRID, (1, _LM_HEAD_K), ttnn.ShardOrientation.ROW_MAJOR),
)


def _gate_bias_target(layer_idx: int) -> TensorTarget:
    return TensorTarget(
        name=f"gate_bias_layer{layer_idx}",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_GATE_BIAS_SENDER_CORE_GRID, (16, 16), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile_shape=(16, 16),
    )


_EMBEDDING_TARGET = TensorTarget(
    name="embedding",
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

_LM_HEAD_TARGET = TensorTarget(
    name="lm_head",
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_LM_HEAD_MATMUL_CORE_GRID, (_LM_HEAD_K, _LM_HEAD_N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
    ),
    mesh_mapper_config=ShardMeshMapper(dim=1),
)

_FINAL_NORM_TARGET = TensorTarget(
    name="final_norm",
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=_NORM_MEM_CONFIG,
    tile_shape=(1, 32),
)


def _mtp_norm_target(name: str) -> TensorTarget:
    return TensorTarget(
        name=name,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_NORM_MEM_CONFIG,
        tile_shape=(1, 32),
    )


def _mtp_eh_proj_target(K: int, N: int) -> TensorTarget:
    n_per_bank = N // _MTP_NUM_DRAM_BANKS
    eh_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_MTP_NUM_DRAM_BANKS - 1, 0))}
    )
    return TensorTarget(
        name="mtp_eh_projection",
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K, n_per_bank), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


def _shared_down_tensor_target(bdw: BlitzDecodeWeights) -> TensorTarget:
    """TensorTarget for shared expert down (L1 WIDTH_SHARDED on matmul cores, bfloat4_b)."""
    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    matmul_core_grid = dp_spec.build_matmul_core_grid()
    dp_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)
    moe_tp = bdw.moe_tp
    if moe_tp == 1:
        mmc = ReplicateMeshMapper()
    else:
        mmc = Shard2dMeshMapper(dims=(0, 1))
    return TensorTarget(
        name="shared_down_proj",
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dp_mem,
        tile_shape=(32, 32),
        mesh_mapper_config=mmc,
    )


def _moe_routed_expert_tensor_target(name: str, K: int, N: int, device) -> TensorTarget:
    """TensorTarget for one MoE routed expert projection (DRAM WIDTH_SHARDED, bfloat4_b)."""
    tile_w = 32
    num_banks = device.dram_grid_size().x
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_N = N_padded // num_banks
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return TensorTarget(
        name=name,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        tile_shape=(32, 32),
        mesh_mapper_config=ReplicateMeshMapper(),
    )


def _dense_routed_stacked_tensor_target(name: str, K: int, N: int, device) -> TensorTarget:
    """TensorTarget for dense MLP routed projection (all experts stacked on mesh)."""
    tile_w = 32
    num_banks = device.dram_grid_size().x
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_N = N_padded // num_banks
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return TensorTarget(
        name=name,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        tile_shape=(32, 32),
        mesh_mapper_config=Shard2dMeshMapper(dims=(0, 1)),
    )


def _dense_mlp_routed_experts_torch(
    mlp_gate: torch.Tensor,
    mlp_up: torch.Tensor,
    mlp_down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same stacked expert tensors as ``get_tt_mlp_routed_expert_weights`` before ``upload``."""
    shared_n = 2048
    num_routed = 8
    expert_n = 2048
    K_gate = mlp_gate.shape[0]
    N_down = mlp_down.shape[1]
    gate_experts = mlp_gate[:, shared_n:].reshape(K_gate, num_routed, expert_n).permute(1, 0, 2).contiguous()
    up_experts = mlp_up[:, shared_n:].reshape(K_gate, num_routed, expert_n).permute(1, 0, 2).contiguous()
    down_experts = mlp_down[shared_n:, :].reshape(num_routed, expert_n, N_down).contiguous()
    return gate_experts, up_experts, down_experts


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
    gate_bias_reshaped = raw_tensor.reshape(16, 16).T.contiguous().to(torch.bfloat16)
    gate_bias_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_GATE_BIAS_SENDER_CORE_GRID, (16, 16), ttnn.ShardOrientation.ROW_MAJOR),
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
_GATE_NUM_INDICES = D.GATE_NUM_INDICES


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
    """Prepare attention fusion groups for one layer (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)."""
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    device = bdw._device

    logger.debug(
        "Converting attention fusion groups for layer {} (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)",
        layer_idx,
    )
    t0 = time.perf_counter()

    q_a_key = _key(layer_idx, "self_attn.q_a_proj.weight")
    q_b_key = _key(layer_idx, "self_attn.q_b_proj.weight")
    kv_a_key = _key(layer_idx, "self_attn.kv_a_proj_with_mqa.weight")
    kv_b_key = _key(layer_idx, "self_attn.kv_b_proj.weight")
    o_proj_key = _key(layer_idx, "self_attn.o_proj.weight")
    attn_norm_key = _key(layer_idx, "input_layernorm.weight")
    q_norm_key = _key(layer_idx, "self_attn.q_a_layernorm.weight")
    kv_norm_key = _key(layer_idx, "self_attn.kv_a_layernorm.weight")
    ffn_norm_key = _key(layer_idx, "post_attention_layernorm.weight")

    def _preprocess_q_ab_kv_a(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q_a = t[q_a_key].T.contiguous()
        q_b = deinterleave_q_b_proj(t[q_b_key])
        kv_a = t[kv_a_key].T.contiguous()
        if bdw.mla_tp == 1 and q_b.shape[1] == _MLA_TP1_Q_B_WIDTH * 2:
            q_b = q_b[:, :_MLA_TP1_Q_B_WIDTH].contiguous()
        return {"q_a_proj": q_a, "q_b_proj": q_b, "kv_a_proj": kv_a}

    q_ab_fp = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=(q_a_key, q_b_key, kv_a_key)),
        target=Q_AB_KV_A_SPEC,
    )
    q_ab_views = cache_config.cache.get_or_create(
        q_ab_fp,
        device,
        preprocess=_preprocess_q_ab_kv_a,
        raw_tensors=lambda: {k: state_dict[k] for k in (q_a_key, q_b_key, kv_a_key)},
    )
    if not isinstance(q_ab_views, dict):
        raise TypeError("expected dict[str, OverlappedTensor] for q_ab_kv_a cache entry")
    q_a_proj = q_ab_views["q_a_proj"]
    q_b_proj = q_ab_views["q_b_proj"]
    kv_a_proj = q_ab_views["kv_a_proj"]

    def _preprocess_kv_b12(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        kv_b1, kv_b2 = split_kv_b_proj(t[kv_b_key])
        if bdw.mla_tp == 1:
            if kv_b1.shape[0] == _MLA_TP1_KV_B1_HEIGHT * 2:
                kv_b1 = kv_b1[:_MLA_TP1_KV_B1_HEIGHT, :].contiguous()
            if kv_b2.shape[1] == _MLA_TP1_KV_B2_WIDTH * 2:
                kv_b2 = kv_b2[:, :_MLA_TP1_KV_B2_WIDTH].contiguous()
        return {"kv_b1_proj": kv_b1, "kv_b2_proj": kv_b2}

    kv_fp = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=(kv_b_key,)),
        target=KV_B12_SPEC,
    )
    kv_views = cache_config.cache.get_or_create(
        kv_fp,
        device,
        preprocess=_preprocess_kv_b12,
        raw_tensors=lambda: {kv_b_key: state_dict[kv_b_key]},
    )
    if not isinstance(kv_views, dict):
        raise TypeError("expected dict[str, OverlappedTensor] for kv_b12 cache entry")
    kv_b1_proj = kv_views["kv_b1_proj"]
    kv_b2_proj = kv_views["kv_b2_proj"]

    if is_moe:
        gate_key = _key(layer_idx, "mlp.gate.weight")

        def _preprocess_o_proj_moe(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            o_proj = t[o_proj_key].T.contiguous()
            gate_mm = t[gate_key].T.contiguous()
            attn_norm = t[attn_norm_key].unsqueeze(0)
            q_norm = t[q_norm_key].unsqueeze(0)
            kv_norm = t[kv_norm_key].unsqueeze(0)
            ffn_norm = t[ffn_norm_key].unsqueeze(0)
            if bdw.mla_tp == 1:
                if o_proj.shape[0] == _MLA_TP1_O_PROJ_HEIGHT * 2:
                    o_proj = o_proj[:_MLA_TP1_O_PROJ_HEIGHT, :].contiguous()
            return {
                "o_proj": o_proj,
                "gate_mm": gate_mm,
                "attn_norm": attn_norm,
                "q_norm": q_norm,
                "kv_norm": kv_norm,
                "ffn_norm": ffn_norm,
            }

        o_src = (o_proj_key, gate_key, attn_norm_key, q_norm_key, kv_norm_key, ffn_norm_key)
        o_fp = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=o_src),
            target=O_PROJ_GATE_MM_NORMS_SPEC,
        )
        o_views = cache_config.cache.get_or_create(
            o_fp,
            device,
            preprocess=_preprocess_o_proj_moe,
            raw_tensors=lambda: {k: state_dict[k] for k in o_src},
        )
        if not isinstance(o_views, dict):
            raise TypeError("expected dict[str, OverlappedTensor] for o_proj_gate_mm_norms cache entry")
        o_proj_ot = o_views["o_proj"]
        gate_mm_ot = o_views["gate_mm"]
        attn_norm_ot = o_views["attn_norm"]
        q_norm_ot = o_views["q_norm"]
        kv_norm_ot = o_views["kv_norm"]
        ffn_norm_ot = o_views["ffn_norm"]

        _bias_key = _key(layer_idx, "mlp.gate.e_score_correction_bias")
        target = _gate_bias_target(layer_idx)
        fingerprint = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(_bias_key,)),
            target=target,
        )
        gate_bias_tt = cache_config.cache.get_or_create(
            fingerprint,
            device,
            preprocess=lambda t: {target.name: t[_bias_key].reshape(16, 16).T.contiguous().to(torch.bfloat16)},
            raw_tensors=lambda: {_bias_key: state_dict[_bias_key]},
        )
        if not isinstance(gate_bias_tt, ttnn.Tensor):
            raise TypeError("expected ttnn.Tensor for gate_bias cache entry")

        logger.debug(
            "Attention fusion groups (MoE) for layer {} in {:.3f}s",
            layer_idx,
            time.perf_counter() - t0,
        )
        return AttentionWeights(
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            o_proj=o_proj_ot,
            gate_mm=gate_mm_ot,
            attn_norm=attn_norm_ot,
            q_norm=q_norm_ot,
            kv_norm=kv_norm_ot,
            ffn_norm=ffn_norm_ot,
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            gate_bias=gate_bias_tt,
        )

    def _preprocess_o_proj_dense(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        o_proj = t[o_proj_key].T.contiguous()
        attn_norm = t[attn_norm_key].unsqueeze(0)
        q_norm = t[q_norm_key].unsqueeze(0)
        kv_norm = t[kv_norm_key].unsqueeze(0)
        ffn_norm = t[ffn_norm_key].unsqueeze(0)
        if bdw.mla_tp == 1 and o_proj.shape[0] == _MLA_TP1_O_PROJ_HEIGHT * 2:
            o_proj = o_proj[:_MLA_TP1_O_PROJ_HEIGHT, :].contiguous()
        gate_mm = torch.zeros(D.HIDDEN_SIZE, D.GATE_NUM_INDICES, dtype=torch.bfloat16, device=o_proj.device)
        return {
            "o_proj": o_proj,
            "gate_mm": gate_mm,
            "attn_norm": attn_norm,
            "q_norm": q_norm,
            "kv_norm": kv_norm,
            "ffn_norm": ffn_norm,
        }

    o_src_dense = (o_proj_key, attn_norm_key, q_norm_key, kv_norm_key, ffn_norm_key)
    o_fp_dense = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=o_src_dense),
        target=O_PROJ_GATE_MM_NORMS_SPEC,
    )
    o_views = cache_config.cache.get_or_create(
        o_fp_dense,
        device,
        preprocess=_preprocess_o_proj_dense,
        raw_tensors=lambda: {k: state_dict[k] for k in o_src_dense},
    )
    if not isinstance(o_views, dict):
        raise TypeError("expected dict[str, OverlappedTensor] for o_proj_gate_mm_norms cache entry")
    o_proj_ot = o_views["o_proj"]
    attn_norm_ot = o_views["attn_norm"]
    q_norm_ot = o_views["q_norm"]
    kv_norm_ot = o_views["kv_norm"]
    ffn_norm_ot = o_views["ffn_norm"]

    logger.debug(
        "Attention fusion groups (dense) for layer {} in {:.3f}s",
        layer_idx,
        time.perf_counter() - t0,
    )
    return AttentionWeights(
        q_a_proj=q_a_proj,
        q_b_proj=q_b_proj,
        kv_a_proj=kv_a_proj,
        o_proj=o_proj_ot,
        gate_mm=None,
        attn_norm=attn_norm_ot,
        q_norm=q_norm_ot,
        kv_norm=kv_norm_ot,
        ffn_norm=ffn_norm_ot,
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
    """Prepare shared expert weights (gate_up fusion group + shared_down_proj) for one layer."""
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    logger.debug("Converting shared expert weights for layer {} (is_moe={})", layer_idx, is_moe)
    t0 = time.perf_counter()
    device = bdw._device
    if is_moe:
        gate_k = _key(layer_idx, "mlp.shared_experts.gate_proj.weight")
        up_k = _key(layer_idx, "mlp.shared_experts.up_proj.weight")
        down_k = _key(layer_idx, "mlp.shared_experts.down_proj.weight")

        def _preprocess_gate_up_moe(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            sg = t[gate_k].T.contiguous()
            su = t[up_k].T.contiguous()
            if bdw.moe_tp == 1:
                full_n = _MOE_TP1_SHARED_GATE_UP_N * 8  # 2048
                if sg.shape[1] == full_n:
                    sg = sg[:, :_MOE_TP1_SHARED_GATE_UP_N].contiguous()
                if su.shape[1] == full_n:
                    su = su[:, :_MOE_TP1_SHARED_GATE_UP_N].contiguous()
            return {"shared_gate_proj": sg, "shared_up_proj": su}

        gu_fp = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(gate_k, up_k)),
            target=GATE_UP_SPEC,
        )
        gu_views = cache_config.cache.get_or_create(
            gu_fp,
            device,
            preprocess=_preprocess_gate_up_moe,
            raw_tensors=lambda: {gate_k: state_dict[gate_k], up_k: state_dict[up_k]},
        )
        if not isinstance(gu_views, dict):
            raise TypeError("expected dict[str, OverlappedTensor] for gate_up cache entry")
        shared_gate_proj = gu_views["shared_gate_proj"]
        shared_up_proj = gu_views["shared_up_proj"]
        sd_target = _shared_down_tensor_target(bdw)
        sd_fp = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(down_k,)),
            target=sd_target,
        )

        def _preprocess_shared_down_moe(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            sd = t[down_k].T.contiguous()
            if bdw.moe_tp == 1 and sd.shape[0] == _MOE_TP1_SHARED_DOWN_K * 8:
                sd = sd[:_MOE_TP1_SHARED_DOWN_K, :].contiguous()
            return {"shared_down_proj": bdw.shared_down_torch_for_cache(sd)}

        shared_down_proj = cache_config.cache.get_or_create(
            sd_fp,
            device,
            preprocess=_preprocess_shared_down_moe,
            raw_tensors=lambda: {down_k: state_dict[down_k]},
        )
        if not isinstance(shared_down_proj, ttnn.Tensor):
            raise TypeError("expected ttnn.Tensor for shared_down_proj cache entry")
    else:
        gate_k = _key(layer_idx, "mlp.gate_proj.weight")
        up_k = _key(layer_idx, "mlp.up_proj.weight")
        down_k = _key(layer_idx, "mlp.down_proj.weight")
        shared_n = 2048

        def _preprocess_gate_up_dense(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            gate = t[gate_k].T.contiguous()
            up = t[up_k].T.contiguous()
            return {
                "shared_gate_proj": gate[:, :shared_n],
                "shared_up_proj": up[:, :shared_n],
            }

        gu_fp = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(gate_k, up_k)),
            target=GATE_UP_SPEC,
        )
        gu_views = cache_config.cache.get_or_create(
            gu_fp,
            device,
            preprocess=_preprocess_gate_up_dense,
            raw_tensors=lambda: {gate_k: state_dict[gate_k], up_k: state_dict[up_k]},
        )
        if not isinstance(gu_views, dict):
            raise TypeError("expected dict[str, OverlappedTensor] for gate_up cache entry")
        shared_gate_proj = gu_views["shared_gate_proj"]
        shared_up_proj = gu_views["shared_up_proj"]
        sd_target = _shared_down_tensor_target(bdw)
        sd_fp = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(down_k,)),
            target=sd_target,
        )

        def _preprocess_shared_down_dense(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            mlp_down = t[down_k].T.contiguous()
            down_slice = mlp_down[:shared_n, :]
            return {"shared_down_proj": bdw.shared_down_torch_for_cache(down_slice)}

        shared_down_proj = cache_config.cache.get_or_create(
            sd_fp,
            device,
            preprocess=_preprocess_shared_down_dense,
            raw_tensors=lambda: {down_k: state_dict[down_k]},
        )
        if not isinstance(shared_down_proj, ttnn.Tensor):
            raise TypeError("expected ttnn.Tensor for shared_down_proj cache entry")
    logger.debug("Shared expert weights done in {:.3f}s", time.perf_counter() - t0)
    return SharedExpertWeights(
        shared_gate_proj=shared_gate_proj,
        shared_up_proj=shared_up_proj,
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
    """Prepare routed expert weights for one layer (dense: single MLP; MoE: num_routed_experts experts)."""
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    device = bdw._device
    if is_moe:
        g0 = state_dict[_key(layer_idx, "mlp.experts.0.gate_proj.weight")].T.contiguous()
        u0 = state_dict[_key(layer_idx, "mlp.experts.0.up_proj.weight")].T.contiguous()
        d0 = state_dict[_key(layer_idx, "mlp.experts.0.down_proj.weight")].T.contiguous()
        Kg, Ng = g0.shape
        Ku, Nu = u0.shape
        Kd, Nd = d0.shape
        tgt_gate = _moe_routed_expert_tensor_target("routed_gate_proj", Kg, Ng, device)
        tgt_up = _moe_routed_expert_tensor_target("routed_up_proj", Ku, Nu, device)
        tgt_down = _moe_routed_expert_tensor_target("routed_down_proj", Kd, Nd, device)
        routed_gate_proj: list[ttnn.Tensor] = []
        routed_up_proj: list[ttnn.Tensor] = []
        routed_down_proj: list[ttnn.Tensor] = []
        for e in range(num_routed_experts):
            gk = _key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")
            fp_g = cache_config.context.fingerprint(
                source=SourceTensorSelection(names=(gk,)),
                target=tgt_gate,
            )
            gw = cache_config.cache.get_or_create(
                fp_g,
                device,
                preprocess=lambda t, _gk=gk: {
                    "routed_gate_proj": bdw.moe_routed_expert_torch_for_cache(t[_gk].T.contiguous())
                },
                raw_tensors=lambda _gk=gk: {_gk: state_dict[_gk]},
            )
            if not isinstance(gw, ttnn.Tensor):
                raise TypeError("expected ttnn.Tensor for routed gate expert cache entry")
            routed_gate_proj.append(gw)
        for e in range(num_routed_experts):
            uk = _key(layer_idx, f"mlp.experts.{e}.up_proj.weight")
            fp_u = cache_config.context.fingerprint(
                source=SourceTensorSelection(names=(uk,)),
                target=tgt_up,
            )
            uw = cache_config.cache.get_or_create(
                fp_u,
                device,
                preprocess=lambda t, _uk=uk: {
                    "routed_up_proj": bdw.moe_routed_expert_torch_for_cache(t[_uk].T.contiguous())
                },
                raw_tensors=lambda _uk=uk: {_uk: state_dict[_uk]},
            )
            if not isinstance(uw, ttnn.Tensor):
                raise TypeError("expected ttnn.Tensor for routed up expert cache entry")
            routed_up_proj.append(uw)
        for e in range(num_routed_experts):
            dk = _key(layer_idx, f"mlp.experts.{e}.down_proj.weight")
            fp_d = cache_config.context.fingerprint(
                source=SourceTensorSelection(names=(dk,)),
                target=tgt_down,
            )
            dw = cache_config.cache.get_or_create(
                fp_d,
                device,
                preprocess=lambda t, _dk=dk: {
                    "routed_down_proj": bdw.moe_routed_expert_torch_for_cache(t[_dk].T.contiguous())
                },
                raw_tensors=lambda _dk=dk: {_dk: state_dict[_dk]},
            )
            if not isinstance(dw, ttnn.Tensor):
                raise TypeError("expected ttnn.Tensor for routed down expert cache entry")
            routed_down_proj.append(dw)
        routed = MoERoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )
        if move_to_device:
            routed.validate_contiguous_dram()
        return routed
    else:
        gate_k = _key(layer_idx, "mlp.gate_proj.weight")
        up_k = _key(layer_idx, "mlp.up_proj.weight")
        down_k = _key(layer_idx, "mlp.down_proj.weight")
        mlp_gate = state_dict[gate_k].T.contiguous()
        mlp_up = state_dict[up_k].T.contiguous()
        mlp_down = state_dict[down_k].T.contiguous()
        gate_experts, up_experts, down_experts = _dense_mlp_routed_experts_torch(mlp_gate, mlp_up, mlp_down)
        K, N = gate_experts.shape[1], gate_experts.shape[2]
        Kd, Nd = down_experts.shape[1], down_experts.shape[2]
        tgt_g = _dense_routed_stacked_tensor_target("routed_gate_proj", K, N, device)
        tgt_u = _dense_routed_stacked_tensor_target("routed_up_proj", K, N, device)
        tgt_d = _dense_routed_stacked_tensor_target("routed_down_proj", Kd, Nd, device)

        fp_g = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(gate_k,)),
            target=tgt_g,
        )

        def _pre_routed_gate(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            mg = t[gate_k].T.contiguous()
            ge = mg[:, 2048:].reshape(mg.shape[0], 8, 2048).permute(1, 0, 2).contiguous()
            return {"routed_gate_proj": bdw.mlp_routed_dense_stacked_torch_for_cache(ge)}

        fp_u = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(up_k,)),
            target=tgt_u,
        )

        def _pre_routed_up(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            mu = t[up_k].T.contiguous()
            ue = mu[:, 2048:].reshape(mu.shape[0], 8, 2048).permute(1, 0, 2).contiguous()
            return {"routed_up_proj": bdw.mlp_routed_dense_stacked_torch_for_cache(ue)}

        fp_d = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(down_k,)),
            target=tgt_d,
        )

        def _pre_routed_down(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            md = t[down_k].T.contiguous()
            de = md[2048:, :].reshape(8, 2048, md.shape[1]).contiguous()
            return {"routed_down_proj": bdw.mlp_routed_dense_stacked_torch_for_cache(de)}

        routed_gate_proj = cache_config.cache.get_or_create(
            fp_g,
            device,
            preprocess=_pre_routed_gate,
            raw_tensors=lambda: {gate_k: state_dict[gate_k]},
        )
        routed_up_proj = cache_config.cache.get_or_create(
            fp_u,
            device,
            preprocess=_pre_routed_up,
            raw_tensors=lambda: {up_k: state_dict[up_k]},
        )
        routed_down_proj = cache_config.cache.get_or_create(
            fp_d,
            device,
            preprocess=_pre_routed_down,
            raw_tensors=lambda: {down_k: state_dict[down_k]},
        )
        if not isinstance(routed_gate_proj, ttnn.Tensor):
            raise TypeError("expected ttnn.Tensor for dense routed_gate_proj cache entry")
        if not isinstance(routed_up_proj, ttnn.Tensor):
            raise TypeError("expected ttnn.Tensor for dense routed_up_proj cache entry")
        if not isinstance(routed_down_proj, ttnn.Tensor):
            raise TypeError("expected ttnn.Tensor for dense routed_down_proj cache entry")
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
        bdw,
        state_dict,
        layer_idx,
        is_moe=False,
        move_to_device=move_to_device,
        cache_config=cache_config,
    )
    shared = prepare_shared_expert_weights(
        bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    routed = prepare_routed_expert_weights(
        bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device, cache_config=cache_config
    )
    assert isinstance(routed, DenseRoutedExpertWeights)
    result = DeepSeekV3DenseLayerWeights(
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
    logger.info("Dense layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)
    return result


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
        bdw,
        state_dict,
        layer_idx,
        is_moe=True,
        move_to_device=move_to_device,
        cache_config=cache_config,
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
    result = DeepSeekV3MoELayerWeights(
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
    logger.info("MoE layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)
    return result


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
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    logger.info("Preparing embedding weights...")
    _src_key = "model.embed_tokens.weight"

    def _preprocess_embedding(t):
        w = t[_src_key]
        assert w.shape == (
            D.VOCAB_SIZE,
            D.HIDDEN_SIZE,
        ), f"Expected embedding shape ({D.VOCAB_SIZE}, {D.HIDDEN_SIZE}), got {w.shape}"
        return {"embedding": w.contiguous()}

    fingerprint = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=(_src_key,)),
        target=_EMBEDDING_TARGET,
    )
    embedding_tt = cache_config.cache.get_or_create(
        fingerprint,
        device,
        preprocess=_preprocess_embedding,
        raw_tensors=lambda: {_src_key: state_dict[_src_key]},
    )
    return DeepSeekV3EmbeddingLayerWeights(embedding=embedding_tt)


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
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    logger.info("Preparing LM head weights...")
    _lm_key = "lm_head.weight"

    def _preprocess_lm_head(t):
        lm_w = t[_lm_key]
        assert lm_w.shape == (
            _LM_HEAD_VOCAB_SIZE,
            _LM_HEAD_K,
        ), f"Expected lm_head shape ({_LM_HEAD_VOCAB_SIZE}, {_LM_HEAD_K}), got {lm_w.shape}"
        return {"lm_head": lm_w.T.contiguous()}

    lm_fingerprint = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=(_lm_key,)),
        target=_LM_HEAD_TARGET,
    )
    lm_head_tt = cache_config.cache.get_or_create(
        lm_fingerprint,
        device,
        preprocess=_preprocess_lm_head,
        raw_tensors=lambda: {_lm_key: state_dict[_lm_key]},
    )

    _norm_key = "model.norm.weight"

    def _preprocess_final_norm(t):
        norm_w = t[_norm_key]
        assert norm_w.shape == (D.HIDDEN_SIZE,), f"Expected final norm shape ({D.HIDDEN_SIZE},), got {norm_w.shape}"
        return {"final_norm": norm_w.unsqueeze(0).contiguous()}

    norm_fingerprint = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=(_norm_key,)),
        target=_FINAL_NORM_TARGET,
    )
    final_norm_tt = cache_config.cache.get_or_create(
        norm_fingerprint,
        device,
        preprocess=_preprocess_final_norm,
        raw_tensors=lambda: {_norm_key: state_dict[_norm_key]},
    )

    return DeepSeekV3LMHeadWeights(lm_head=lm_head_tt, final_norm=final_norm_tt)


def _transform_eh_proj(eh_proj_weight_T: torch.Tensor) -> torch.Tensor:
    """Pad to DRAM bank alignment and tile-shuffle. Input: already transposed (K, N)."""
    K, N = eh_proj_weight_T.shape
    assert N % _MTP_NUM_DRAM_BANKS == 0, f"eh_proj N={N} must be divisible by {_MTP_NUM_DRAM_BANKS} DRAM banks"
    n_per_bank = N // _MTP_NUM_DRAM_BANKS
    padded_N = _MTP_NUM_DRAM_BANKS * n_per_bank
    eh_padded = torch.zeros((K, padded_N), dtype=eh_proj_weight_T.dtype)
    eh_padded[:, :N] = eh_proj_weight_T
    return BlitzDecodeWeights._shuffle_dram_tiles(eh_padded, 32, _MTP_NUM_DRAM_BANKS).contiguous()


def _to_tt_mtp_eh_proj(eh_proj_torch: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Convert transposed eh_proj (K+embedding_dim, hidden) to TT WIDTH_SHARDED DRAM with tile shuffle.

    eh_proj_torch: already transposed to (K, N) = (14336, 7168).
    Pads N to align with _MTP_NUM_DRAM_BANKS, shuffles tiles for DRAM streaming,
    and creates a WIDTH_SHARDED DRAM tensor.
    """
    K, N = eh_proj_torch.shape
    n_per_bank = N // _MTP_NUM_DRAM_BANKS
    eh_shuffled = _transform_eh_proj(eh_proj_torch)

    eh_shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(_MTP_NUM_DRAM_BANKS - 1, 0),
            )
        }
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K, n_per_bank), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        eh_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device if move_to_device else None,
        memory_config=eh_proj_mem_config,
        tile=_MTP_B_TILE,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _mtp_eh_proj_preprocess(raw: dict[str, torch.Tensor], src_key: str, target_name: str) -> dict[str, torch.Tensor]:
    """Preprocess eh_proj for cache: transpose, pad to DRAM bank alignment, tile-shuffle."""
    return {target_name: _transform_eh_proj(raw[src_key].T.contiguous())}


def prepare_mtp_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    *,
    mtp_layer_idx: int = _MTP_LAYER_IDX,
    move_to_device: bool = False,
    cache_config: CacheConfig | None = None,
) -> DeepSeekV3MTPWeights:
    """Prepare lightweight MTP projection/norm weights from state dict.

    Only the MTP-specific tensors (h_gamma, e_gamma, eh_projection) are prepared here.
    The MTP decoder block (layer 61) is a regular MoE layer handled through ``prepare_moe_layer_weights``.
    """
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    logger.info("Preparing MTP weights (layer {})...", mtp_layer_idx)
    t0 = time.perf_counter()

    _h_key = _key(mtp_layer_idx, "hnorm.weight")
    h_target = _mtp_norm_target("mtp_h_gamma")
    h_fingerprint = cache_config.context.fingerprint(source=SourceTensorSelection(names=(_h_key,)), target=h_target)
    h_gamma_tt = cache_config.cache.get_or_create(
        h_fingerprint,
        device,
        preprocess=lambda t: {h_target.name: t[_h_key].unsqueeze(0).contiguous()},
        raw_tensors=lambda: {_h_key: state_dict[_h_key]},
    )

    _e_key = _key(mtp_layer_idx, "enorm.weight")
    e_target = _mtp_norm_target("mtp_e_gamma")
    e_fingerprint = cache_config.context.fingerprint(source=SourceTensorSelection(names=(_e_key,)), target=e_target)
    e_gamma_tt = cache_config.cache.get_or_create(
        e_fingerprint,
        device,
        preprocess=lambda t: {e_target.name: t[_e_key].unsqueeze(0).contiguous()},
        raw_tensors=lambda: {_e_key: state_dict[_e_key]},
    )

    _eh_key = _key(mtp_layer_idx, "eh_proj.weight")
    eh_target = _mtp_eh_proj_target(K=2 * _LM_HEAD_K, N=_LM_HEAD_K)
    eh_fingerprint = cache_config.context.fingerprint(source=SourceTensorSelection(names=(_eh_key,)), target=eh_target)
    eh_proj_tt = cache_config.cache.get_or_create(
        eh_fingerprint,
        device,
        preprocess=lambda t: _mtp_eh_proj_preprocess(t, _eh_key, eh_target.name),
        raw_tensors=lambda: {_eh_key: state_dict[_eh_key]},
    )

    logger.info("MTP weights prepared in {:.3f}s", time.perf_counter() - t0)
    return DeepSeekV3MTPWeights(
        h_gamma=h_gamma_tt,
        e_gamma=e_gamma_tt,
        eh_projection=eh_proj_tt,
    )
