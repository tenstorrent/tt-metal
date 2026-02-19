# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Provides prepare_weights(state_dict, device, ...) -> DeepSeekV3Weights and
dataclasses for dense vs MoE layer weight containers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    BlitzDecodeWeights,
    OverlappedTensor,
)


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

    # From get_tt_kv_b12_proj_weights
    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor

    # From get_tt_gate_up_proj_weights (shared expert)
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor


DeepSeekV3LayerWeights = DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights


@dataclass
class DeepSeekV3Weights:
    """Container for all prepared (fused) layer weights."""

    layers: list[DeepSeekV3LayerWeights]


# Constants for kv_b_proj split (HF stores one matrix; we split into kv_b1 and kv_b2).
_NUM_HEADS = 64
_QK_NOPE_HEAD_DIM = 128
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256


def _key(layer_idx: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_idx}."""
    return f"model.layers.{layer_idx}.{suffix}"


def _split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split HF kv_b_proj (out_features, in_features) into kv_b1 (8192, 512) and kv_b2 (512, 8192).

    HF shape: (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank) = (16384, 512).
    Reshape to (64, 256, 512); first 128 of the 256 dim are k (b1), last 128 are v (b2).
    Only kv_b2 is transposed for blitz; kv_b1 is used as (8192, 512).
    """
    # (16384, 512) -> (64, 256, 512); same layout as reference mla1d (num_heads, head_dim, kv_lora_rank)
    w = kv_b_proj.reshape(_NUM_HEADS, _KV_B_PROJ_HEAD_DIM, _KV_LORA_RANK).contiguous()
    # kv_b1: (64, 128, 512) -> (8192, 512); no transpose (blitz expects (8192, 512))
    kv_b1 = w[:, : _QK_NOPE_HEAD_DIM, :].reshape(-1, _KV_LORA_RANK)
    # kv_b2: (64, 128, 512) -> (8192, 512) then transpose -> (512, 8192) for blitz
    kv_b2 = w[:, _QK_NOPE_HEAD_DIM :, :].reshape(-1, _KV_LORA_RANK).T.contiguous()
    return kv_b1, kv_b2


def prepare_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    num_layers: int = 61,
    first_k_dense_replace: int = 3,
) -> DeepSeekV3Weights:
    """Build fused (blitz decode) weights from a HuggingFace-style state dict.

    Args:
        state_dict: Weights keyed by model.layers.{i}.self_attn.*, model.layers.{i}.mlp.*, etc.
        device: ttnn device (or MeshDevice) to place tensors on.
        num_layers: Total number of layers (default 61).
        first_k_dense_replace: Number of dense layers before MoE (default 3).

    Returns:
        DeepSeekV3Weights with one entry per layer; dense vs MoE type by layer index.
    """
    bdw = BlitzDecodeWeights(device)
    layers: list[DeepSeekV3LayerWeights] = []

    for i in range(num_layers):
        prefix = f"model.layers.{i}."

        # Linear weights: HF (out_features, in_features) -> (K, N) for blitz. All transposed except kv_b1.
        q_a = state_dict[_key(i, "self_attn.q_a_proj.weight")].T.contiguous()
        q_b = state_dict[_key(i, "self_attn.q_b_proj.weight")].T.contiguous()
        kv_a = state_dict[_key(i, "self_attn.kv_a_proj_with_mqa.weight")].T.contiguous()
        kv_b1, kv_b2 = _split_kv_b_proj(state_dict[_key(i, "self_attn.kv_b_proj.weight")])  # b1 no transpose, b2 transposed inside
        o_proj = state_dict[_key(i, "self_attn.o_proj.weight")].T.contiguous()

        # Norms: (C,) -> (1, C)
        attn_norm = state_dict[_key(i, "input_layernorm.weight")].unsqueeze(0)
        q_norm = state_dict[_key(i, "self_attn.q_a_layernorm.weight")].unsqueeze(0)
        kv_norm = state_dict[_key(i, "self_attn.kv_a_layernorm.weight")].unsqueeze(0)
        ffn_norm = state_dict[_key(i, "post_attention_layernorm.weight")].unsqueeze(0)

        q_ab_kv_a = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a, q_b, kv_a)
        q_a_proj, q_b_proj, kv_a_proj = q_ab_kv_a

        kv_b12 = bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2)
        kv_b1_proj, kv_b2_proj = kv_b12

        if i < first_k_dense_replace:
            gate_mm_dummy = torch.zeros(
                7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device
            )
            o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
                o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm
            )
            o_proj_ot, gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
            layers.append(
                DeepSeekV3DenseLayerWeights(
                    q_a_proj=q_a_proj,
                    q_b_proj=q_b_proj,
                    kv_a_proj=kv_a_proj,
                    o_proj=o_proj_ot,
                    attn_norm=attn_norm_ot,
                    q_norm=q_norm_ot,
                    kv_norm=kv_norm_ot,
                    ffn_norm=ffn_norm_ot,
                    kv_b1_proj=kv_b1_proj,
                    kv_b2_proj=kv_b2_proj,
                )
            )
        else:
            gate_mm = state_dict[_key(i, "mlp.gate.weight")].T.contiguous()
            shared_gate = state_dict[_key(i, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
            shared_up = state_dict[_key(i, "mlp.shared_experts.up_proj.weight")].T.contiguous()

            o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
                o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm
            )
            o_proj_ot, gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
            gate_up = bdw.get_tt_gate_up_proj_weights(shared_gate, shared_up)
            shared_gate_proj, shared_up_proj = gate_up

            layers.append(
                DeepSeekV3MoELayerWeights(
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
                    shared_gate_proj=shared_gate_proj,
                    shared_up_proj=shared_up_proj,
                )
            )

    return DeepSeekV3Weights(layers=layers)
