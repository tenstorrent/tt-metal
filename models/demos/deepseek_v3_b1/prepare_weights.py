# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (full logical shapes for the target
mesh), applies key mapping, transpose, and kv_b split, then passes to
BlitzDecodeWeights which fuses and shards onto the mesh.

Supports per-layer save/load (save_decoder_layer, load_dense_decoder_layer, load_moe_decoder_layer),
embedding/lm_head save/load, and MTP weight save/load for offline preparation and runtime load.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights, OverlappedTensor

# Serialization: manifest version and dtype name mapping
_MANIFEST_VERSION = 1
_DTYPE_TO_STR = {
    ttnn.DataType.BFLOAT4_B: "BFLOAT4_B",
    ttnn.DataType.BFLOAT8_B: "BFLOAT8_B",
    ttnn.DataType.UINT32: "UINT32",
    ttnn.DataType.BFLOAT16: "BFLOAT16",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}

# MoE sender core: hardcoded grid (13, 10) so cache layout is consistent across slow/fast dispatch.
# Sender core = (grid.x - 1, grid.y - 1) = (12, 9); must match test_moe_mlp create_runtime_tensors.
MOE_SENDER_GRID_SIZE = (13, 10)
_GATE_BIAS_TILE = ttnn.Tile([16, 16])

# Fusion group name per field (for grouping by fused_tensor)
_FIELD_TO_FUSION_GROUP: dict[str, str] = {
    "q_a_proj": "q_ab_kv_a",
    "q_b_proj": "q_ab_kv_a",
    "kv_a_proj": "q_ab_kv_a",
    "o_proj": "o_proj_gate_mm_norms",
    "gate_mm": "o_proj_gate_mm_norms",
    "attn_norm": "o_proj_gate_mm_norms",
    "q_norm": "o_proj_gate_mm_norms",
    "kv_norm": "o_proj_gate_mm_norms",
    "ffn_norm": "o_proj_gate_mm_norms",
    "kv_b1_proj": "kv_b12",
    "kv_b2_proj": "kv_b12",
    "shared_gate_proj": "gate_up",
    "shared_up_proj": "gate_up",
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
NUM_ROUTED_EXPERTS = 256
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256
_Q_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192

# MTP layer constants
_MTP_LAYER_IDX = 61
_MTP_NUM_DRAM_BANKS = 8
_MTP_B_TILE = ttnn.Tile([32, 32])


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
) -> AttentionWeights:
    """Prepare attention fusion groups for one layer (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)."""
    logger.debug("Loading raw tensors from state dict for layer {}", layer_idx)
    t0 = time.perf_counter()
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = get_layer_raw_tensors(
        state_dict, layer_idx
    )
    # Single-device (mla_tp=1) expects per-TP shapes; slice if state dict has full logical (2-TP) size
    q_b, o_proj, kv_b1, kv_b2 = _slice_attention_weights_for_mla_tp(q_b, o_proj, kv_b1, kv_b2, bdw.mla_tp)
    logger.debug("Loaded raw tensors in {:.3f}s", time.perf_counter() - t0)
    logger.debug("Converting attention fusion groups for layer {} (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)", layer_idx)
    t0 = time.perf_counter()
    q_a_proj, q_b_proj, kv_a_proj = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(
        q_a, q_b, kv_a, move_to_device=move_to_device
    )
    kv_b1_proj, kv_b2_proj = bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2, move_to_device=move_to_device)
    logger.debug("Converted q_ab_kv_a + kv_b12 in {:.3f}s", time.perf_counter() - t0)

    if is_moe:
        gate_mm = state_dict[_key(layer_idx, "mlp.gate.weight")].T.contiguous()
        o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
            o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
        )
        o_proj_ot, gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
        gate_bias_tt = create_gate_bias_tensor(
            state_dict[_key(layer_idx, "mlp.gate.e_score_correction_bias")],
            bdw._device,
            move_to_device=move_to_device,
        )
        logger.debug("Converted o_proj_gate_mm_norms (MoE) in {:.3f}s", time.perf_counter() - t0)
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
    else:
        gate_mm_dummy = torch.zeros(7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device)
        o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
            o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
        )
        o_proj_ot, _gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
        logger.debug("Converted o_proj_gate_mm_norms (dense) in {:.3f}s", time.perf_counter() - t0)
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
) -> SharedExpertWeights:
    """Prepare shared expert weights (gate_up fusion group + shared_down_proj) for one layer."""
    logger.debug("Converting shared expert weights for layer {} (is_moe={})", layer_idx, is_moe)
    t0 = time.perf_counter()
    if is_moe:
        shared_gate = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
        shared_up = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()
        shared_down = state_dict[_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T.contiguous()
        # Single-device (moe_tp=1) expects per-TP shapes; slice if state dict has full logical (8-TP) size
        shared_gate, shared_up, shared_down = _slice_shared_expert_weights_for_moe_tp(
            shared_gate, shared_up, shared_down, bdw.moe_tp
        )
        shared_gate_proj, shared_up_proj, shared_down_proj = bdw.get_tt_moe_shared_expert_weights(
            shared_gate, shared_up, shared_down, move_to_device=move_to_device
        )
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()
        shared_gate_proj, shared_up_proj, shared_down_proj = bdw.get_tt_mlp_shared_expert_weights(
            mlp_gate, mlp_up, mlp_down, move_to_device=move_to_device
        )
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
) -> DenseRoutedExpertWeights | MoERoutedExpertWeights:
    """Prepare routed expert weights for one layer (dense: single MLP; MoE: num_routed_experts experts)."""
    if is_moe:
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
                logger.debug("Loaded experts 0..{} from state dict", e - 1)
            gate_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")].T.contiguous())
            up_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")].T.contiguous())
            down_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")].T.contiguous())
        load_elapsed = time.perf_counter() - t0
        logger.info("Loaded {} experts from state dict in {:.3f}s", num_routed_experts, load_elapsed)
        logger.debug("Converting routed experts to device format (blitz)...")
        t0 = time.perf_counter()
        gate_stacked = torch.stack(gate_list, dim=0)
        up_stacked = torch.stack(up_list, dim=0)
        down_stacked = torch.stack(down_list, dim=0)
        routed_gate_proj, routed_up_proj, routed_down_proj = bdw.get_tt_moe_routed_expert_weights(
            gate_stacked, up_stacked, down_stacked, move_to_device=move_to_device
        )
        logger.info("Converted routed experts in {:.3f}s", time.perf_counter() - t0)
        routed = MoERoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )
        if move_to_device:
            routed.validate_contiguous_dram()
        return routed
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()
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
) -> DeepSeekV3DenseLayerWeights:
    """Prepare fused weights for a single dense decoder layer."""
    logger.info("Preparing dense layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device)
    routed = prepare_routed_expert_weights(bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device)
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
) -> DeepSeekV3MoELayerWeights:
    """Prepare fused weights for a single MoE decoder layer."""
    logger.info("Preparing MoE layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=move_to_device)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=move_to_device)
    routed = prepare_routed_expert_weights(
        bdw, state_dict, layer_idx, is_moe=True, num_routed_experts=num_routed_experts, move_to_device=move_to_device
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
) -> DeepSeekV3EmbeddingLayerWeights:
    """Prepare embedding weights from state dict (model.embed_tokens.weight)."""
    logger.info("Preparing embedding weights...")
    w = state_dict["model.embed_tokens.weight"]
    assert w.shape == (129280, 7168), f"Expected embedding shape (129280, 7168), got {w.shape}"
    embedding_tt = _to_tt_embedding(w, device, move_to_device=move_to_device)
    return DeepSeekV3EmbeddingLayerWeights(embedding=embedding_tt)


def save_embedding_weights(
    weights: DeepSeekV3EmbeddingLayerWeights,
    path: str | Path,
    *,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save embedding weights to <path>/embedding/."""
    path = Path(path)
    emb_dir = path / "embedding"
    emb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving embedding weights...")
    ttnn.dump_tensor(emb_dir / "embedding.tensorbin", weights.embedding)
    logger.info("Saving embedding manifest...")
    manifest = {
        "version": _MANIFEST_VERSION,
        "hf_model_name": hf_model_name,
        "hf_state_dict_name": hf_state_dict_name,
        "device_mesh_shape": list(device_mesh_shape),
    }
    with open(emb_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_embedding_weights(path: str | Path, device) -> DeepSeekV3EmbeddingLayerWeights:
    """Load embedding weights from <path>/embedding/."""
    path = Path(path)
    emb_dir = path / "embedding"
    if not emb_dir.is_dir():
        raise FileNotFoundError(f"Embedding dir not found: {emb_dir}")
    embedding = ttnn.load_tensor(emb_dir / "embedding.tensorbin", device=device)
    return DeepSeekV3EmbeddingLayerWeights(embedding=embedding)


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
) -> DeepSeekV3LMHeadWeights:
    """Prepare LM head and final norm weights from state dict.

    device must be the mesh device (e.g. 4x2 submesh). The LM head weight matrix is sharded
    along the vocabulary dimension (TP = mesh size). Per-device layout matches the LM head
    sampling op: WIDTH_SHARDED in L1 across 101 matmul cores with shard shape (7168, N_per_core).
    """
    logger.info("Preparing LM head weights...")
    # lm_head.weight: HF (vocab_size, hidden_size) = (129280, 7168) -> (7168, 129280) for matmul
    lm_w = state_dict["lm_head.weight"]
    assert lm_w.shape == (
        _LM_HEAD_VOCAB_SIZE,
        _LM_HEAD_K,
    ), f"Expected lm_head shape ({_LM_HEAD_VOCAB_SIZE}, {_LM_HEAD_K}), got {lm_w.shape}"

    lm_head_tt = _to_tt_lm_head_matrix(
        lm_w.T, device, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1), move_to_device=move_to_device
    )

    # model.norm.weight: (7168,) -> (1, 7168), HEIGHT_SHARDED on the mcast core
    logger.info("Preparing LM head norm...")
    norm_w = state_dict["model.norm.weight"]
    assert norm_w.shape == (7168,), f"Expected final norm shape (7168,), got {norm_w.shape}"

    final_norm_tt = _to_tt_lm_head_final_norm(norm_w.unsqueeze(0), device, move_to_device=move_to_device)
    return DeepSeekV3LMHeadWeights(lm_head=lm_head_tt, final_norm=final_norm_tt)


def save_lm_head_weights(
    weights: DeepSeekV3LMHeadWeights,
    path: str | Path,
    *,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save LM head and final norm weights to <path>/lm_head/."""
    path = Path(path)
    lm_dir = path / "lm_head"
    lm_dir.mkdir(parents=True, exist_ok=True)
    ttnn.dump_tensor(lm_dir / "lm_head.tensorbin", weights.lm_head)
    ttnn.dump_tensor(lm_dir / "final_norm.tensorbin", weights.final_norm)
    manifest = {
        "version": _MANIFEST_VERSION,
        "hf_model_name": hf_model_name,
        "hf_state_dict_name": hf_state_dict_name,
        "device_mesh_shape": list(device_mesh_shape),
    }
    with open(lm_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_lm_head_weights(path: str | Path, device) -> DeepSeekV3LMHeadWeights:
    """Load LM head and final norm weights from <path>/lm_head/.

    device must be the mesh device (same shape as used for prepare_lm_head_weights) so the
    loaded LM head has the same vocab-dim sharding (TP = mesh size).
    """
    path = Path(path)
    lm_dir = path / "lm_head"
    if not lm_dir.is_dir():
        raise FileNotFoundError(f"LM head dir not found: {lm_dir}")
    lm_head = ttnn.load_tensor(lm_dir / "lm_head.tensorbin", device=device)
    final_norm = ttnn.load_tensor(lm_dir / "final_norm.tensorbin", device=device)
    return DeepSeekV3LMHeadWeights(lm_head=lm_head, final_norm=final_norm)


def _to_tt_mtp_eh_proj(eh_proj_torch: torch.Tensor, device, *, move_to_device: bool = False) -> ttnn.Tensor:
    """Convert transposed eh_proj (K+embedding_dim, hidden) to TT WIDTH_SHARDED DRAM with tile shuffle.

    eh_proj_torch: already transposed to (K, N) = (14336, 7168).
    Pads N to align with _MTP_NUM_DRAM_BANKS, shuffles tiles for DRAM streaming,
    and creates a WIDTH_SHARDED DRAM tensor.
    """
    K, N = eh_proj_torch.shape
    assert N % _MTP_NUM_DRAM_BANKS == 0, f"eh_proj N={N} must be divisible by {_MTP_NUM_DRAM_BANKS} DRAM banks"
    n_per_bank = N // _MTP_NUM_DRAM_BANKS
    padded_N = _MTP_NUM_DRAM_BANKS * n_per_bank

    eh_padded = torch.zeros((K, padded_N), dtype=eh_proj_torch.dtype)
    eh_padded[:, :N] = eh_proj_torch

    eh_shuffled = BlitzDecodeWeights._shuffle_dram_tiles(eh_padded, 32, _MTP_NUM_DRAM_BANKS)

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
        eh_shuffled.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device if move_to_device else None,
        memory_config=eh_proj_mem_config,
        tile=_MTP_B_TILE,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def prepare_mtp_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    *,
    mtp_layer_idx: int = _MTP_LAYER_IDX,
    move_to_device: bool = False,
) -> DeepSeekV3MTPWeights:
    """Prepare lightweight MTP projection/norm weights from state dict.

    Only the MTP-specific tensors (h_gamma, e_gamma, eh_projection) are prepared here.
    The MTP decoder block (layer 61) is a regular MoE layer handled through the standard
    prepare_moe_layer_weights / load_moe_decoder_layer path.
    """
    logger.info("Preparing MTP weights (layer {})...", mtp_layer_idx)
    t0 = time.perf_counter()

    h_gamma_w = state_dict[_key(mtp_layer_idx, "hnorm.weight")].unsqueeze(0)
    h_gamma_tt = _to_tt_lm_head_final_norm(h_gamma_w, device, move_to_device=move_to_device)

    e_gamma_w = state_dict[_key(mtp_layer_idx, "enorm.weight")].unsqueeze(0)
    e_gamma_tt = _to_tt_lm_head_final_norm(e_gamma_w, device, move_to_device=move_to_device)

    eh_proj_w = state_dict[_key(mtp_layer_idx, "eh_proj.weight")].T.contiguous()
    eh_proj_tt = _to_tt_mtp_eh_proj(eh_proj_w, device, move_to_device=move_to_device)

    logger.info("MTP weights prepared in {:.3f}s", time.perf_counter() - t0)
    return DeepSeekV3MTPWeights(
        h_gamma=h_gamma_tt,
        e_gamma=e_gamma_tt,
        eh_projection=eh_proj_tt,
    )


def save_mtp_weights(
    weights: DeepSeekV3MTPWeights,
    path: str | Path,
    *,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save MTP projection/norm weights to <path>/mtp/.

    Only the lightweight MTP tensors (h_gamma, e_gamma, eh_projection) are saved here.
    The MTP decoder block (layer 61) is saved separately as a standard MoE layer via
    save_decoder_layer.
    """
    path = Path(path)
    mtp_dir = path / "mtp"
    mtp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving MTP projection/norm weights to {}...", mtp_dir)
    t0 = time.perf_counter()
    ttnn.dump_tensor(mtp_dir / "mtp_h_gamma.tensorbin", weights.h_gamma)
    ttnn.dump_tensor(mtp_dir / "mtp_e_gamma.tensorbin", weights.e_gamma)
    ttnn.dump_tensor(mtp_dir / "mtp_eh_projection.tensorbin", weights.eh_projection)
    manifest = {
        "version": _MANIFEST_VERSION,
        "hf_model_name": hf_model_name,
        "hf_state_dict_name": hf_state_dict_name,
        "device_mesh_shape": list(device_mesh_shape),
    }
    with open(mtp_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("MTP weights saved in {:.3f}s", time.perf_counter() - t0)


def load_mtp_weights(
    path: str | Path,
    device,
) -> DeepSeekV3MTPWeights:
    """Load MTP projection/norm weights from <path>/mtp/.

    device must be the mesh device (same shape as used for prepare_mtp_weights).
    The MTP decoder block (layer 61) is loaded separately via load_moe_decoder_layer.
    """
    path = Path(path)
    mtp_dir = path / "mtp"
    if not mtp_dir.is_dir():
        raise FileNotFoundError(f"MTP dir not found: {mtp_dir}")
    logger.info("Loading MTP weights from {}...", mtp_dir)
    t0 = time.perf_counter()
    h_gamma = ttnn.load_tensor(mtp_dir / "mtp_h_gamma.tensorbin", device=device)
    e_gamma = ttnn.load_tensor(mtp_dir / "mtp_e_gamma.tensorbin", device=device)
    eh_projection = ttnn.load_tensor(mtp_dir / "mtp_eh_projection.tensorbin", device=device)
    logger.info("MTP weights loaded in {:.3f}s", time.perf_counter() - t0)
    return DeepSeekV3MTPWeights(
        h_gamma=h_gamma,
        e_gamma=e_gamma,
        eh_projection=eh_projection,
    )


def _core_range_set_to_list(crs: ttnn.CoreRangeSet) -> list[list[list[int]]]:
    """Serialize CoreRangeSet to JSON-serializable list of [[sx, sy], [ex, ey]]."""
    result = []
    for r in crs.ranges():
        start, end = r.start, r.end
        result.append([[start.x, start.y], [end.x, end.y]])
    return result


def _core_range_set_from_list(lst: list[list[list[int]]]) -> ttnn.CoreRangeSet:
    """Deserialize list of [[sx, sy], [ex, ey]] to CoreRangeSet."""
    ranges = [
        ttnn.CoreRange(
            ttnn.CoreCoord(pair[0][0], pair[0][1]),
            ttnn.CoreCoord(pair[1][0], pair[1][1]),
        )
        for pair in lst
    ]
    return ttnn.CoreRangeSet(ranges)


_OVERLAPPED_SERIALIZED_FIELDS = {
    "tensor_shape",
    "shard_shape",
    "core_range_set",
    "dtype",
    "tile_shape",
    "byte_offset",
    "total_size",
}
_OVERLAPPED_SKIP_FIELDS = {"fused_tensor"}


def _overlapped_tensor_to_json(ot: OverlappedTensor) -> dict:
    """Serialize one OverlappedTensor's metadata to a JSON-serializable dict."""
    all_fields = {f.name for f in fields(OverlappedTensor)}
    missing = all_fields - _OVERLAPPED_SERIALIZED_FIELDS - _OVERLAPPED_SKIP_FIELDS
    assert not missing, f"OverlappedTensor has new fields not serialized: {missing}"
    dtype_str = _DTYPE_TO_STR.get(ot.dtype)
    if dtype_str is None:
        dtype_str = str(ot.dtype)
    return {
        "tensor_shape": list(ot.tensor_shape),
        "shard_shape": list(ot.shard_shape),
        "core_range_set": _core_range_set_to_list(ot.core_range_set),
        "dtype": dtype_str,
        "tile_shape": list(ot.tile_shape),
        "byte_offset": ot.byte_offset,
        "total_size": ot.total_size,
    }


def _overlapped_tensor_from_dict(
    fused_tensor: ttnn.Tensor,
    d: dict,
) -> OverlappedTensor:
    """Reconstruct one OverlappedTensor from loaded fused tensor and manifest dict."""
    dtype = _STR_TO_DTYPE.get(d["dtype"])
    if dtype is None:
        raise ValueError(f"Unknown dtype in manifest: {d['dtype']}")
    total_size = d.get("total_size", 0)
    if total_size == 0:
        raise ValueError("manifest is missing 'total_size' for OverlappedTensor — regenerate the weight cache")
    return OverlappedTensor(
        fused_tensor=fused_tensor,
        tensor_shape=tuple(d["tensor_shape"]),
        shard_shape=tuple(d["shard_shape"]),
        core_range_set=_core_range_set_from_list(d["core_range_set"]),
        dtype=dtype,
        tile_shape=tuple(d["tile_shape"]),
        byte_offset=d["byte_offset"],
        total_size=total_size,
    )


def _layer_overlapped_tensor_fields(
    layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights,
) -> list[tuple[str, OverlappedTensor]]:
    """Return (field_name, OverlappedTensor) for every OverlappedTensor field on the layer."""
    out = []
    for f in fields(layer):
        val = getattr(layer, f.name)
        if isinstance(val, OverlappedTensor):
            out.append((f.name, val))
    return out


def _read_or_create_manifest(
    layer_dir: Path,
    layer_idx: int,
    is_moe: bool,
    *,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> dict:
    """Read existing manifest or create a new one for incremental save. Caller merges and writes back."""
    manifest_path = layer_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "version": _MANIFEST_VERSION,
        "created_time": created,
        "hf_model_name": hf_model_name,
        "hf_state_dict_name": hf_state_dict_name,
        "device_mesh_shape": list(device_mesh_shape),
        "layer_idx": layer_idx,
        "layer_type": "moe" if is_moe else "dense",
        "fusion_groups": {},
        "standalone_tensors": {},
    }
    if is_moe:
        manifest["routed_experts"] = {"num_experts": NUM_ROUTED_EXPERTS}
    else:
        manifest["routed_mlp"] = True
    return manifest


def _dump_overlapped_fusion_groups(
    layer_dir: Path,
    field_tuples: list[tuple[str, OverlappedTensor]],
) -> dict:
    """Dump fused tensors for the given (field_name, OverlappedTensor) pairs; return fusion_groups dict."""
    by_fused: dict[int, list[tuple[str, OverlappedTensor]]] = {}
    for name, ot in field_tuples:
        fid = id(ot.fused_tensor)
        if fid not in by_fused:
            by_fused[fid] = []
        by_fused[fid].append((name, ot))
    fusion_groups: dict[str, dict] = {}
    for fid, group_fields in by_fused.items():
        group_name = _FIELD_TO_FUSION_GROUP.get(group_fields[0][0])
        if group_name is None:
            raise KeyError(f"Unknown field for fusion group: {group_fields[0][0]}")
        tensorbin_name = f"{group_name}.tensorbin"
        ttnn.dump_tensor(layer_dir / tensorbin_name, group_fields[0][1].fused_tensor)
        fusion_groups[group_name] = {
            "tensorbin": tensorbin_name,
            "fields": {name: _overlapped_tensor_to_json(ot) for name, ot in group_fields},
        }
    return fusion_groups


def save_attention_weights(
    attn: AttentionWeights,
    path: str | Path,
    layer_idx: int,
    *,
    is_moe: bool,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save attention fusion groups to layer dir; merge into existing manifest if present."""
    logger.debug("Saving attention weights for layer {}...", layer_idx)
    t0 = time.perf_counter()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    layer_dir = path / f"layer_{layer_idx:03d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_or_create_manifest(
        layer_dir,
        layer_idx,
        is_moe,
        hf_model_name=hf_model_name,
        hf_state_dict_name=hf_state_dict_name,
        device_mesh_shape=device_mesh_shape,
    )
    field_tuples: list[tuple[str, OverlappedTensor]] = [
        ("q_a_proj", attn.q_a_proj),
        ("q_b_proj", attn.q_b_proj),
        ("kv_a_proj", attn.kv_a_proj),
        ("o_proj", attn.o_proj),
        ("attn_norm", attn.attn_norm),
        ("q_norm", attn.q_norm),
        ("kv_norm", attn.kv_norm),
        ("ffn_norm", attn.ffn_norm),
        ("kv_b1_proj", attn.kv_b1_proj),
        ("kv_b2_proj", attn.kv_b2_proj),
    ]
    if attn.gate_mm is not None:
        field_tuples.append(("gate_mm", attn.gate_mm))
    new_groups = _dump_overlapped_fusion_groups(layer_dir, field_tuples)
    manifest.setdefault("fusion_groups", {}).update(new_groups)
    if attn.gate_bias is not None:
        ttnn.dump_tensor(layer_dir / "gate_bias.tensorbin", attn.gate_bias)
        manifest.setdefault("standalone_tensors", {})["gate_bias"] = "gate_bias.tensorbin"
    with open(layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.debug("Saved attention weights in {:.3f}s", time.perf_counter() - t0)


def save_shared_expert_weights(
    shared: SharedExpertWeights,
    path: str | Path,
    layer_idx: int,
    *,
    is_moe: bool,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save shared expert gate_up and shared_down_proj to layer dir; merge into existing manifest if present."""
    logger.debug("Saving shared expert weights for layer {}...", layer_idx)
    t0 = time.perf_counter()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    layer_dir = path / f"layer_{layer_idx:03d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_or_create_manifest(
        layer_dir,
        layer_idx,
        is_moe,
        hf_model_name=hf_model_name,
        hf_state_dict_name=hf_state_dict_name,
        device_mesh_shape=device_mesh_shape,
    )
    new_groups = _dump_overlapped_fusion_groups(
        layer_dir,
        [
            ("shared_gate_proj", shared.shared_gate_proj),
            ("shared_up_proj", shared.shared_up_proj),
        ],
    )
    manifest.setdefault("fusion_groups", {}).update(new_groups)
    name = "shared_down_proj.tensorbin"
    ttnn.dump_tensor(layer_dir / name, shared.shared_down_proj)
    manifest.setdefault("standalone_tensors", {})["shared_down_proj"] = name
    with open(layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.debug("Saved shared expert weights in {:.3f}s", time.perf_counter() - t0)


def save_routed_expert_weights(
    routed: DenseRoutedExpertWeights | MoERoutedExpertWeights,
    path: str | Path,
    layer_idx: int,
    *,
    is_moe: bool,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save routed expert weights to layer dir; merge into existing manifest if present."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    layer_dir = path / f"layer_{layer_idx:03d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_or_create_manifest(
        layer_dir,
        layer_idx,
        is_moe,
        hf_model_name=hf_model_name,
        hf_state_dict_name=hf_state_dict_name,
        device_mesh_shape=device_mesh_shape,
    )
    standalone = manifest.setdefault("standalone_tensors", {})
    if is_moe:
        assert isinstance(routed, MoERoutedExpertWeights)
        num_experts = len(routed.routed_gate_proj)
        manifest["routed_experts"] = {"num_experts": num_experts}
        logger.info("Saving {} routed experts for layer {} to disk (this may be slow)...", num_experts, layer_idx)
        t0 = time.perf_counter()
        experts_dir = layer_dir / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)
        for e in range(num_experts):
            if e > 0 and e % 64 == 0:
                logger.debug("Saved experts 0..{}", e - 1)
            expert_dir = experts_dir / f"e_{e:03d}"
            expert_dir.mkdir(parents=True, exist_ok=True)
            ttnn.dump_tensor(expert_dir / "gate_proj.tensorbin", routed.routed_gate_proj[e])
            ttnn.dump_tensor(expert_dir / "up_proj.tensorbin", routed.routed_up_proj[e])
            ttnn.dump_tensor(expert_dir / "down_proj.tensorbin", routed.routed_down_proj[e])
        logger.info("Saved {} routed experts in {:.3f}s", num_experts, time.perf_counter() - t0)
    else:
        assert isinstance(routed, DenseRoutedExpertWeights)
        logger.debug("Saving dense routed MLP for layer {}...", layer_idx)
        ttnn.dump_tensor(layer_dir / "routed_gate_proj.tensorbin", routed.routed_gate_proj)
        ttnn.dump_tensor(layer_dir / "routed_up_proj.tensorbin", routed.routed_up_proj)
        ttnn.dump_tensor(layer_dir / "routed_down_proj.tensorbin", routed.routed_down_proj)
        standalone["routed_gate_proj"] = "routed_gate_proj.tensorbin"
        standalone["routed_up_proj"] = "routed_up_proj.tensorbin"
        standalone["routed_down_proj"] = "routed_down_proj.tensorbin"
    with open(layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def save_decoder_layer(
    layer: DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights,
    path: str | Path,
    layer_idx: int,
    *,
    hf_model_name: str,
    hf_state_dict_name: str,
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Serialize a single layer to <path>/layer_{layer_idx:03d}/.

    Creates one directory with manifest.json and per-fusion-group .tensorbin files.
    Caller must provide hf_model_name and hf_state_dict_name for the manifest.
    """
    path = Path(path)
    layer_dir = path / f"layer_{layer_idx:03d}"
    logger.info("Saving layer {} to {}...", layer_idx, layer_dir)
    is_moe = isinstance(layer, DeepSeekV3MoELayerWeights)
    save_decoder_layer_t0 = time.perf_counter()
    attn = AttentionWeights(
        q_a_proj=layer.q_a_proj,
        q_b_proj=layer.q_b_proj,
        kv_a_proj=layer.kv_a_proj,
        o_proj=layer.o_proj,
        gate_mm=getattr(layer, "gate_mm", None),
        attn_norm=layer.attn_norm,
        q_norm=layer.q_norm,
        kv_norm=layer.kv_norm,
        ffn_norm=layer.ffn_norm,
        kv_b1_proj=layer.kv_b1_proj,
        kv_b2_proj=layer.kv_b2_proj,
        gate_bias=getattr(layer, "gate_bias", None),
    )
    shared = SharedExpertWeights(
        shared_gate_proj=layer.shared_gate_proj,
        shared_up_proj=layer.shared_up_proj,
        shared_down_proj=layer.shared_down_proj,
    )
    if is_moe:
        routed = MoERoutedExpertWeights(
            routed_gate_proj=layer.routed_gate_proj,
            routed_up_proj=layer.routed_up_proj,
            routed_down_proj=layer.routed_down_proj,
        )
    else:
        routed = DenseRoutedExpertWeights(
            routed_gate_proj=layer.routed_gate_proj,
            routed_up_proj=layer.routed_up_proj,
            routed_down_proj=layer.routed_down_proj,
        )
    save_attention_weights(
        attn,
        path,
        layer_idx,
        is_moe=is_moe,
        hf_model_name=hf_model_name,
        hf_state_dict_name=hf_state_dict_name,
        device_mesh_shape=device_mesh_shape,
    )
    save_shared_expert_weights(
        shared,
        path,
        layer_idx,
        is_moe=is_moe,
        hf_model_name=hf_model_name,
        hf_state_dict_name=hf_state_dict_name,
        device_mesh_shape=device_mesh_shape,
    )
    save_routed_expert_weights(
        routed,
        path,
        layer_idx,
        is_moe=is_moe,
        hf_model_name=hf_model_name,
        hf_state_dict_name=hf_state_dict_name,
        device_mesh_shape=device_mesh_shape,
    )
    logger.info("Saved decoder layer in {:.3f}s", time.perf_counter() - save_decoder_layer_t0)


def load_moe_routed_experts(
    path: str | Path,
    device,
    layer_idx: int,
    *,
    num_experts: int = NUM_ROUTED_EXPERTS,
) -> MoERoutedExpertWeights:
    """Load only the routed expert weights for an MoE layer from cache.

    Reads experts/e_NNN/{gate,up,down}_proj.tensorbin. Loads all gate experts, then all
    up, then all down, so DRAM allocation order matches ``get_tt_moe_routed_expert_weights``
    (required for DRAMStreamingMatmul expert indexing).

    Since setup_fast_dispatch can only be used once per program, call this under
    setup_fast_dispatch and pass the result to load_moe_decoder_layer(...,
    preloaded_routed_experts=...). If you do not use fast dispatch, omit
    preloaded_routed_experts and load_moe_decoder_layer will load experts from disk in the
    current dispatch mode.
    """
    path = Path(path)
    layer_dir = path / f"layer_{layer_idx:03d}"
    manifest_path = layer_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("layer_type") != "moe":
        raise ValueError(f"Layer {layer_idx} is not MoE (layer_type={manifest.get('layer_type')})")
    num_experts = manifest.get("routed_experts", {}).get("num_experts", num_experts)
    experts_dir = layer_dir / "experts"
    logger.info("Loading {} routed experts for layer {} from cache...", num_experts, layer_idx)
    t0 = time.perf_counter()
    routed_gate_proj = []
    routed_up_proj = []
    routed_down_proj = []
    for e in range(num_experts):
        if e > 0 and e % 64 == 0:
            logger.debug("Loaded gate experts 0..{}", e - 1)
        expert_dir = experts_dir / f"e_{e:03d}"
        routed_gate_proj.append(ttnn.load_tensor(expert_dir / "gate_proj.tensorbin", device=device))
    for e in range(num_experts):
        if e > 0 and e % 64 == 0:
            logger.debug("Loaded up experts 0..{}", e - 1)
        expert_dir = experts_dir / f"e_{e:03d}"
        routed_up_proj.append(ttnn.load_tensor(expert_dir / "up_proj.tensorbin", device=device))
    for e in range(num_experts):
        if e > 0 and e % 64 == 0:
            logger.debug("Loaded down experts 0..{}", e - 1)
        expert_dir = experts_dir / f"e_{e:03d}"
        routed_down_proj.append(ttnn.load_tensor(expert_dir / "down_proj.tensorbin", device=device))

    logger.info("Routed experts for layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - t0)
    routed = MoERoutedExpertWeights(
        routed_gate_proj=routed_gate_proj,
        routed_up_proj=routed_up_proj,
        routed_down_proj=routed_down_proj,
    )
    routed.validate_contiguous_dram()
    return routed


def load_dense_decoder_layer(
    path: str | Path,
    device,
    layer_idx: int,
) -> DeepSeekV3DenseLayerWeights:
    """Deserialize a dense decoder layer from <path>/layer_{layer_idx:03d}/."""
    path = Path(path)
    layer_dir = path / f"layer_{layer_idx:03d}"
    manifest_path = layer_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("version", 0) > _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")

    if manifest.get("layer_type") != "dense":
        raise ValueError(f"Layer {layer_idx} is not dense (layer_type={manifest.get('layer_type')})")

    fusion_groups = manifest["fusion_groups"]
    load_t0 = time.perf_counter()
    logger.info("Loading layer {} (dense) from disk...", layer_idx)

    q_ab = fusion_groups["q_ab_kv_a"]
    fused_q = ttnn.load_tensor(layer_dir / q_ab["tensorbin"], device=device)
    q_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_a_proj"])
    q_b_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_b_proj"])
    kv_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["kv_a_proj"])

    o_grp = fusion_groups["o_proj_gate_mm_norms"]
    fused_o = ttnn.load_tensor(layer_dir / o_grp["tensorbin"], device=device)
    o_proj = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["o_proj"])
    attn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["attn_norm"])
    q_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["q_norm"])
    kv_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["kv_norm"])
    ffn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["ffn_norm"])

    kv_grp = fusion_groups["kv_b12"]
    fused_kv = ttnn.load_tensor(layer_dir / kv_grp["tensorbin"], device=device)
    kv_b1_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b1_proj"])
    kv_b2_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b2_proj"])

    gu_grp = fusion_groups["gate_up"]
    fused_gu = ttnn.load_tensor(layer_dir / gu_grp["tensorbin"], device=device)
    shared_gate_proj = _overlapped_tensor_from_dict(fused_gu, gu_grp["fields"]["shared_gate_proj"])
    shared_up_proj = _overlapped_tensor_from_dict(fused_gu, gu_grp["fields"]["shared_up_proj"])

    standalone = manifest.get("standalone_tensors", {})
    shared_down_proj = ttnn.load_tensor(layer_dir / standalone["shared_down_proj"], device=device)
    routed_gate_proj = ttnn.load_tensor(layer_dir / standalone["routed_gate_proj"], device=device)
    routed_up_proj = ttnn.load_tensor(layer_dir / standalone["routed_up_proj"], device=device)
    routed_down_proj = ttnn.load_tensor(layer_dir / standalone["routed_down_proj"], device=device)
    logger.info("Layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - load_t0)

    return DeepSeekV3DenseLayerWeights(
        q_a_proj=q_a_proj,
        q_b_proj=q_b_proj,
        kv_a_proj=kv_a_proj,
        o_proj=o_proj,
        attn_norm=attn_norm,
        q_norm=q_norm,
        kv_norm=kv_norm,
        ffn_norm=ffn_norm,
        kv_b1_proj=kv_b1_proj,
        kv_b2_proj=kv_b2_proj,
        shared_gate_proj=shared_gate_proj,
        shared_up_proj=shared_up_proj,
        shared_down_proj=shared_down_proj,
        routed_gate_proj=routed_gate_proj,
        routed_up_proj=routed_up_proj,
        routed_down_proj=routed_down_proj,
    )


def load_moe_decoder_layer(
    path: str | Path,
    device,
    layer_idx: int,
    *,
    preloaded_routed_experts: MoERoutedExpertWeights | None = None,
) -> DeepSeekV3MoELayerWeights:
    """Deserialize an MoE decoder layer from <path>/layer_{layer_idx:03d}/.

    If preloaded_routed_experts is provided (e.g. from load_moe_routed_experts under
    setup_fast_dispatch, which can only be used once per program), those experts are
    used. Otherwise routed experts are loaded from disk in the current dispatch mode.
    Fusion groups and standalone tensors are always loaded in the current dispatch mode.
    """
    path = Path(path)
    layer_dir = path / f"layer_{layer_idx:03d}"
    manifest_path = layer_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("version", 0) > _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")

    if manifest.get("layer_type") != "moe":
        raise ValueError(f"Layer {layer_idx} is not MoE (layer_type={manifest.get('layer_type')})")

    fusion_groups = manifest["fusion_groups"]
    load_t0 = time.perf_counter()
    logger.info("Loading layer {} (moe) from disk...", layer_idx)

    if preloaded_routed_experts is None:
        preloaded_routed_experts = load_moe_routed_experts(path, device, layer_idx)

    q_ab = fusion_groups["q_ab_kv_a"]
    fused_q = ttnn.load_tensor(layer_dir / q_ab["tensorbin"], device=device)
    q_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_a_proj"])
    q_b_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["q_b_proj"])
    kv_a_proj = _overlapped_tensor_from_dict(fused_q, q_ab["fields"]["kv_a_proj"])

    o_grp = fusion_groups["o_proj_gate_mm_norms"]
    fused_o = ttnn.load_tensor(layer_dir / o_grp["tensorbin"], device=device)
    o_proj = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["o_proj"])
    gate_mm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["gate_mm"])
    attn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["attn_norm"])
    q_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["q_norm"])
    kv_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["kv_norm"])
    ffn_norm = _overlapped_tensor_from_dict(fused_o, o_grp["fields"]["ffn_norm"])

    kv_grp = fusion_groups["kv_b12"]
    fused_kv = ttnn.load_tensor(layer_dir / kv_grp["tensorbin"], device=device)
    kv_b1_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b1_proj"])
    kv_b2_proj = _overlapped_tensor_from_dict(fused_kv, kv_grp["fields"]["kv_b2_proj"])

    gu_grp = fusion_groups["gate_up"]
    fused_gu = ttnn.load_tensor(layer_dir / gu_grp["tensorbin"], device=device)
    shared_gate_proj = _overlapped_tensor_from_dict(fused_gu, gu_grp["fields"]["shared_gate_proj"])
    shared_up_proj = _overlapped_tensor_from_dict(fused_gu, gu_grp["fields"]["shared_up_proj"])

    standalone = manifest.get("standalone_tensors", {})
    shared_down_proj = ttnn.load_tensor(layer_dir / standalone["shared_down_proj"], device=device)
    gate_bias = ttnn.load_tensor(layer_dir / standalone["gate_bias"], device=device)
    routed_gate_proj = preloaded_routed_experts.routed_gate_proj
    routed_up_proj = preloaded_routed_experts.routed_up_proj
    routed_down_proj = preloaded_routed_experts.routed_down_proj
    logger.info("Layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - load_t0)

    return DeepSeekV3MoELayerWeights(
        q_a_proj=q_a_proj,
        q_b_proj=q_b_proj,
        kv_a_proj=kv_a_proj,
        o_proj=o_proj,
        gate_mm=gate_mm,
        attn_norm=attn_norm,
        q_norm=q_norm,
        kv_norm=kv_norm,
        ffn_norm=ffn_norm,
        gate_bias=gate_bias,
        kv_b1_proj=kv_b1_proj,
        kv_b2_proj=kv_b2_proj,
        shared_gate_proj=shared_gate_proj,
        shared_up_proj=shared_up_proj,
        shared_down_proj=shared_down_proj,
        routed_gate_proj=routed_gate_proj,
        routed_up_proj=routed_up_proj,
        routed_down_proj=routed_down_proj,
    )
