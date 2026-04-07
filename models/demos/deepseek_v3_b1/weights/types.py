# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-specific assembled weight dataclasses and MoE layout helpers.

These types are the **Assemble** layer output: high-level per-layer structures built from
cached **artifacts** (standalone tensors and fusion groups). Generic tensor-cache types live
under :mod:`models.demos.deepseek_v3_b1.tensor_cache`.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D


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

    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor
    o_proj: OverlappedTensor
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor
    routed_gate_proj: ttnn.Tensor
    routed_up_proj: ttnn.Tensor
    routed_down_proj: ttnn.Tensor


@dataclass
class DeepSeekV3MoELayerWeights:
    """Weights for an MoE layer (first_k_dense_replace..num_layers-1).

    Extends dense with gate_mm and shared expert projections.
    """

    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor
    o_proj: OverlappedTensor
    gate_mm: OverlappedTensor
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor
    gate_bias: ttnn.Tensor
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor
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

    h_gamma: ttnn.Tensor
    e_gamma: ttnn.Tensor
    eh_projection: ttnn.Tensor


# MoE routed experts (DeepSeek V3 config: n_routed_experts=256).
NUM_ROUTED_EXPERTS = D.GATE_NUM_INDICES
_ROUTED_GATE_UP_K = D.HIDDEN_SIZE  # 7168
_ROUTED_GATE_UP_N = D.MOE_INTERMEDIATE_SIZE  # 2048
_ROUTED_DOWN_K = D.MOE_INTERMEDIATE_SIZE  # 2048
_ROUTED_DOWN_N = D.HIDDEN_SIZE  # 7168
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_KV_LORA_RANK = D.KV_B_LORA_RANK
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256
_Q_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192

# MTP layer constants
_MTP_LAYER_IDX = 61
_MTP_NUM_DRAM_BANKS = 8
