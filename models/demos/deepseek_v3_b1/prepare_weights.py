# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (full logical shapes for the target
mesh), applies key mapping, transpose, and kv_b split, then passes to
BlitzDecodeWeights which fuses and shards onto the mesh.

Supports save_weights / load_weights for offline preparation and runtime load.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

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


@dataclass
class MoERoutedExpertWeights:
    """Routed expert weights for MoE layers (list of tensors, one per expert)."""

    routed_gate_proj: list[ttnn.Tensor]
    routed_up_proj: list[ttnn.Tensor]
    routed_down_proj: list[ttnn.Tensor]


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


DeepSeekV3LayerWeights = DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights


@dataclass
class DeepSeekV3Weights:
    """Container for all prepared (fused) layer weights."""

    layers: list[DeepSeekV3LayerWeights]


# Constants for kv_b_proj split (HF stores one matrix; we split into kv_b1 and kv_b2).
_NUM_HEADS = 64
# MoE routed experts (DeepSeek V3 config: n_routed_experts=256).
NUM_ROUTED_EXPERTS = 256
_QK_NOPE_HEAD_DIM = 128
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256


def _key(layer_idx: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_idx}."""
    return f"model.layers.{layer_idx}.{suffix}"


def _split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _get_layer_raw_tensors(
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
    (1, W); kv_b_proj is split into kv_b1 and kv_b2 (see _split_kv_b_proj).

    Transformation (HF full logical -> transform -> passed to BlitzDecodeWeights):

        Weight        | HF key (under model.layers.{i}.)     | HF shape      | Transform   | To blitz
        --------------|-------------------------------------|---------------|-------------|------------------
        q_b_proj      | self_attn.q_b_proj.weight            | (24576, 1536) | .T          | (1536, 24576)
        o_proj        | self_attn.o_proj.weight              | (7168, 16384) | .T          | (16384, 7168)
        kv_b_proj     | self_attn.kv_b_proj.weight           | (32768, 512)  | split       | kv_b1, kv_b2
        q_a_proj      | self_attn.q_a_proj.weight            | (1536, 7168)  | .T          | (7168, 1536)
        kv_a_proj     | self_attn.kv_a_proj_with_mqa.weight  | (576, 7168)   | .T          | (7168, 576)
        norms         | input_layernorm, q_a_layernorm, etc. | (7168,), …    | unsqueeze(0)| (1, 7168), …

    MoE-only (gate_mm, shared_gate_proj, shared_up_proj) are read in
    prepare_moe_decoder_layer_weights.

    Returns:
        (q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm).
    """
    q_a = state_dict[_key(layer_idx, "self_attn.q_a_proj.weight")].T.contiguous()
    q_b = state_dict[_key(layer_idx, "self_attn.q_b_proj.weight")].T.contiguous()
    kv_a = state_dict[_key(layer_idx, "self_attn.kv_a_proj_with_mqa.weight")].T.contiguous()
    kv_b1, kv_b2 = _split_kv_b_proj(state_dict[_key(layer_idx, "self_attn.kv_b_proj.weight")])
    o_proj = state_dict[_key(layer_idx, "self_attn.o_proj.weight")].T.contiguous()

    attn_norm = state_dict[_key(layer_idx, "input_layernorm.weight")].unsqueeze(0)
    q_norm = state_dict[_key(layer_idx, "self_attn.q_a_layernorm.weight")].unsqueeze(0)
    kv_norm = state_dict[_key(layer_idx, "self_attn.kv_a_layernorm.weight")].unsqueeze(0)
    ffn_norm = state_dict[_key(layer_idx, "post_attention_layernorm.weight")].unsqueeze(0)

    return q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm


def prepare_attention_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
) -> AttentionWeights:
    """Prepare attention fusion groups for one layer (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)."""
    logger.debug("Loading raw tensors from state dict for layer {}", layer_idx)
    t0 = time.perf_counter()
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = _get_layer_raw_tensors(
        state_dict, layer_idx
    )
    logger.debug("  load raw tensors: {:.3f}s", time.perf_counter() - t0)
    logger.debug("Converting attention fusion groups for layer {} (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)", layer_idx)
    t0 = time.perf_counter()
    q_a_proj, q_b_proj, kv_a_proj = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a, q_b, kv_a, move_to_device=False)
    kv_b1_proj, kv_b2_proj = bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2, move_to_device=False)
    logger.debug("  convert q_ab_kv_a + kv_b12: {:.3f}s", time.perf_counter() - t0)

    if is_moe:
        gate_mm = state_dict[_key(layer_idx, "mlp.gate.weight")].T.contiguous()
        o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
            o_proj, gate_mm, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=False
        )
        o_proj_ot, gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
        logger.debug("  convert o_proj_gate_mm_norms (MoE): {:.3f}s", time.perf_counter() - t0)
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
        )
    else:
        gate_mm_dummy = torch.zeros(7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device)
        o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
            o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=False
        )
        o_proj_ot, _gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
        logger.debug("  convert o_proj_gate_mm_norms (dense): {:.3f}s", time.perf_counter() - t0)
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
        )


def prepare_shared_expert_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
) -> SharedExpertWeights:
    """Prepare shared expert weights (gate_up fusion group + shared_down_proj) for one layer."""
    logger.debug("Converting shared expert weights for layer {} (is_moe={})", layer_idx, is_moe)
    t0 = time.perf_counter()
    if is_moe:
        shared_gate = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
        shared_up = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()
        shared_down = state_dict[_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T.contiguous()
        shared_gate_proj, shared_up_proj, shared_down_proj = bdw.get_tt_moe_shared_expert_weights(
            shared_gate, shared_up, shared_down, move_to_device=False
        )
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()
        shared_gate_proj, shared_up_proj, shared_down_proj = bdw.get_tt_mlp_shared_expert_weights(
            mlp_gate, mlp_up, mlp_down, move_to_device=False
        )
    logger.debug("  shared expert weights done in {:.3f}s", time.perf_counter() - t0)
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
            gate_stacked, up_stacked, down_stacked, move_to_device=False
        )
        logger.info("  converted routed experts in {:.3f}s", time.perf_counter() - t0)
        return MoERoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()
        routed_gate_proj, routed_up_proj, routed_down_proj = bdw.get_tt_mlp_routed_expert_weights(
            mlp_gate, mlp_up, mlp_down, move_to_device=False
        )
        return DenseRoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )


def prepare_dense_decoder_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
) -> DeepSeekV3DenseLayerWeights:
    """Prepare fused weights for a single dense decoder layer."""
    logger.info("Preparing dense layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=False)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=False)
    routed = prepare_routed_expert_weights(bdw, state_dict, layer_idx, is_moe=False)
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


def prepare_moe_decoder_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> DeepSeekV3MoELayerWeights:
    """Prepare fused weights for a single MoE decoder layer."""
    logger.info("Preparing MoE layer {}...", layer_idx)
    t0 = time.perf_counter()
    attn = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=True)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True)
    routed = prepare_routed_expert_weights(
        bdw, state_dict, layer_idx, is_moe=True, num_routed_experts=num_routed_experts
    )
    assert isinstance(attn.gate_mm, OverlappedTensor)
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


def prepare_weights(
    state_dict: dict[str, torch.Tensor],
    device,
    num_layers: int = 61,
    first_k_dense_replace: int = 3,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> DeepSeekV3Weights:
    """Build fused weights from a HuggingFace-style state dict.

    State dict should use full logical HF shapes when device is a mesh (e.g. 4x2);
    internally we shard them across the mesh.

    Args:
        state_dict: Weights keyed by model.layers.{i}.self_attn.*, model.layers.{i}.mlp.*, etc.
        device: MeshDevice to place weights on.
        num_layers: Total number of layers (default 61).
        first_k_dense_replace: Number of dense layers before MoE (default 3).
        num_routed_experts: Number of MoE routed experts per layer (default NUM_ROUTED_EXPERTS).

    Returns:
        DeepSeekV3Weights with one entry per layer; dense vs MoE type by layer index.
    """
    logger.info(
        "Preparing full model weights ({} layers, dense 0..{}, MoE {}..{})...",
        num_layers,
        first_k_dense_replace - 1,
        first_k_dense_replace,
        num_layers - 1,
    )
    total_t0 = time.perf_counter()
    bdw = BlitzDecodeWeights(device)
    layers: list[DeepSeekV3LayerWeights] = []

    for i in range(num_layers):
        is_moe = i >= first_k_dense_replace
        if is_moe:
            layers.append(prepare_moe_decoder_layer_weights(bdw, state_dict, i, num_routed_experts=num_routed_experts))
        else:
            layers.append(prepare_dense_decoder_layer_weights(bdw, state_dict, i))

    logger.info("All {} layers prepared in {:.3f}s", num_layers, time.perf_counter() - total_t0)
    return DeepSeekV3Weights(layers=layers)


def _deallocate_tt_tensor(t: ttnn.Tensor, seen: set[int]) -> None:
    """Deallocate a single ttnn.Tensor if not already deallocated (by id)."""
    fid = id(t)
    if fid not in seen:
        seen.add(fid)
        ttnn.deallocate(t, force=True)


def deallocate_weights(weights: DeepSeekV3Weights) -> None:
    """Release device memory for all fused tensors in prepared weights.

    Call this before loading a new set of weights onto the same device to avoid
    OOM (the original and loaded weights would otherwise both reside on device).
    """
    seen: set[int] = set()
    for layer in weights.layers:
        for _name, ot in _layer_overlapped_tensor_fields(layer):
            fid = id(ot.fused_tensor)
            if fid not in seen:
                seen.add(fid)
                ttnn.deallocate(ot.fused_tensor, force=True)
        if hasattr(layer, "shared_down_proj"):
            _deallocate_tt_tensor(getattr(layer, "shared_down_proj"), seen)
        if isinstance(layer, DeepSeekV3MoELayerWeights):
            for t in layer.routed_gate_proj:
                _deallocate_tt_tensor(t, seen)
            for t in layer.routed_up_proj:
                _deallocate_tt_tensor(t, seen)
            for t in layer.routed_down_proj:
                _deallocate_tt_tensor(t, seen)
        else:
            assert isinstance(layer, DeepSeekV3DenseLayerWeights)
            _deallocate_tt_tensor(layer.routed_gate_proj, seen)
            _deallocate_tt_tensor(layer.routed_up_proj, seen)
            _deallocate_tt_tensor(layer.routed_down_proj, seen)


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


def _overlapped_tensor_to_json(ot: OverlappedTensor) -> dict:
    """Serialize one OverlappedTensor's metadata to a JSON-serializable dict."""
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
    }


def _overlapped_tensor_from_dict(
    fused_tensor: ttnn.Tensor,
    d: dict,
) -> OverlappedTensor:
    """Reconstruct one OverlappedTensor from loaded fused tensor and manifest dict."""
    dtype = _STR_TO_DTYPE.get(d["dtype"])
    if dtype is None:
        raise ValueError(f"Unknown dtype in manifest: {d['dtype']}")
    return OverlappedTensor(
        fused_tensor=fused_tensor,
        tensor_shape=tuple(d["tensor_shape"]),
        shard_shape=tuple(d["shard_shape"]),
        core_range_set=_core_range_set_from_list(d["core_range_set"]),
        dtype=dtype,
        tile_shape=tuple(d["tile_shape"]),
        byte_offset=d["byte_offset"],
    )


def _layer_overlapped_tensor_fields(
    layer: DeepSeekV3LayerWeights,
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
    with open(layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.debug("  save_attention_weights: {:.3f}s", time.perf_counter() - t0)


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
    logger.debug("  save_shared_expert_weights: {:.3f}s", time.perf_counter() - t0)


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
                logger.debug("  saved experts 0..{}", e - 1)
            expert_dir = experts_dir / f"e_{e:03d}"
            expert_dir.mkdir(parents=True, exist_ok=True)
            ttnn.dump_tensor(expert_dir / "gate_proj.tensorbin", routed.routed_gate_proj[e])
            ttnn.dump_tensor(expert_dir / "up_proj.tensorbin", routed.routed_up_proj[e])
            ttnn.dump_tensor(expert_dir / "down_proj.tensorbin", routed.routed_down_proj[e])
        logger.info("  saved {} routed experts in {:.3f}s", num_experts, time.perf_counter() - t0)
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


def save_layer(
    layer: DeepSeekV3LayerWeights,
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
    logger.info(f"Saving layer {layer_idx} to {layer_dir}...")
    is_moe = isinstance(layer, DeepSeekV3MoELayerWeights)
    save_layer_t0 = time.perf_counter()
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
    logger.info(f"  save_layer total: {time.perf_counter() - save_layer_t0:.3f}s")


def load_layer(
    path: str | Path,
    device,
    layer_idx: int,
) -> DeepSeekV3LayerWeights:
    """Deserialize a single layer from <path>/layer_{layer_idx:03d}/."""
    path = Path(path)
    layer_dir = path / f"layer_{layer_idx:03d}"
    manifest_path = layer_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("version", 0) > _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")

    layer_type = manifest["layer_type"]
    fusion_groups = manifest["fusion_groups"]
    load_t0 = time.perf_counter()
    logger.info("Loading layer {} ({}) from disk...", layer_idx, layer_type)

    if layer_type == "dense":
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
        logger.info("  layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - load_t0)

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
    else:
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
        num_experts = manifest.get("routed_experts", {}).get("num_experts", NUM_ROUTED_EXPERTS)
        logger.info("  loading {} routed experts from disk (this may be slow)...", num_experts)
        experts_t0 = time.perf_counter()
        experts_dir = layer_dir / "experts"
        routed_gate_proj = []
        routed_up_proj = []
        routed_down_proj = []
        for e in range(num_experts):
            if e > 0 and e % 64 == 0:
                logger.debug("  loaded experts 0..{}", e - 1)
            expert_dir = experts_dir / f"e_{e:03d}"
            routed_gate_proj.append(ttnn.load_tensor(expert_dir / "gate_proj.tensorbin", device=device))
            routed_up_proj.append(ttnn.load_tensor(expert_dir / "up_proj.tensorbin", device=device))
            routed_down_proj.append(ttnn.load_tensor(expert_dir / "down_proj.tensorbin", device=device))
        logger.info(
            "  layer {} loaded in {:.3f}s (routed experts: {:.3f}s)",
            layer_idx,
            time.perf_counter() - load_t0,
            time.perf_counter() - experts_t0,
        )

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
            kv_b1_proj=kv_b1_proj,
            kv_b2_proj=kv_b2_proj,
            shared_gate_proj=shared_gate_proj,
            shared_up_proj=shared_up_proj,
            shared_down_proj=shared_down_proj,
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )


def save_weights(
    weights: DeepSeekV3Weights,
    path: str | Path,
    *,
    hf_model_name: str,
    hf_state_dict_name: str,
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Serialize all layers to disk. Convenience wrapper around save_layer."""
    path = Path(path)
    num_layers = len(weights.layers)
    logger.info("Saving all {} layers to {}...", num_layers, path)
    total_t0 = time.perf_counter()
    for layer_idx, layer in enumerate(weights.layers):
        save_layer(
            layer,
            path,
            layer_idx,
            hf_model_name=hf_model_name,
            hf_state_dict_name=hf_state_dict_name,
            device_mesh_shape=device_mesh_shape,
        )
    logger.info("All {} layers saved in {:.3f}s", num_layers, time.perf_counter() - total_t0)


def load_weights(
    path: str | Path,
    device,
    num_layers: int = 61,
) -> DeepSeekV3Weights:
    """Deserialize layers from disk. Convenience wrapper: load_layer for each index."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Weights path is not a directory: {path}")
    logger.info("Loading all {} layers from {}...", num_layers, path)
    total_t0 = time.perf_counter()
    layers = [load_layer(path, device, i) for i in range(num_layers)]
    logger.info("All {} layers loaded in {:.3f}s", num_layers, time.perf_counter() - total_t0)
    return DeepSeekV3Weights(layers=layers)
