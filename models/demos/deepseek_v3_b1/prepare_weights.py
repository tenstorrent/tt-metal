# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare DeepSeek V3 fused (blitz decode) weights from a state dict.

Takes full HuggingFace state dict tensors (full logical shapes for the target
mesh), applies key mapping, transpose, and kv_b split, then passes to
BlitzDecodeWeights which fuses and shards onto the mesh.

Supports per-layer save/load (save_decoder_layer, load_dense_decoder_layer, load_moe_decoder_layer) and embedding/lm_head save/load for offline preparation and runtime load.
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
    """Attention fusion groups: q_ab_kv_a + kv_b12 + o_proj_gate_mm_norms (no gate routing)."""

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


@dataclass
class GateWeights:
    """MoE gate routing weights."""

    gate_mm: OverlappedTensor
    gate_bias: ttnn.Tensor


@dataclass
class SharedExpertWeights:
    """Shared expert gate_up fusion group + standalone shared_down_proj."""

    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor


@dataclass
class RoutedExpertWeights:
    """Routed expert weights — list of tensors per projection (one per expert).

    Dense: 1 element per list (single mesh-sharded tensor).
    MoE: 256 elements per list (one ttnn.Tensor per expert).
    Kernel uses list[0].buffer_address() as base and strides by per-expert size.
    """

    routed_gate_proj: list[ttnn.Tensor]
    routed_up_proj: list[ttnn.Tensor]
    routed_down_proj: list[ttnn.Tensor]


@dataclass
class DeepSeekV3DenseLayerWeights:
    """Weights for a dense layer (0..first_k_dense_replace-1)."""

    attention: AttentionWeights
    shared_expert: SharedExpertWeights
    routed_expert: RoutedExpertWeights


@dataclass
class DeepSeekV3MoELayerWeights:
    """Weights for an MoE layer (first_k_dense_replace..num_layers-1)."""

    attention: AttentionWeights
    gate: GateWeights
    shared_expert: SharedExpertWeights
    routed_expert: RoutedExpertWeights


DeepSeekV3DecoderLayerWeights = DeepSeekV3MoELayerWeights | DeepSeekV3DenseLayerWeights


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
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256


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
    prepare_moe_layer_weights.

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
    move_to_device: bool = False,
) -> tuple[AttentionWeights, GateWeights | None]:
    """Prepare attention fusion groups for one layer (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms).
    Returns (AttentionWeights, GateWeights | None). GateWeights is non-None only for MoE layers.
    """
    logger.debug("Loading raw tensors from state dict for layer {}", layer_idx)
    t0 = time.perf_counter()
    q_a, q_b, kv_a, kv_b1, kv_b2, o_proj, attn_norm, q_norm, kv_norm, ffn_norm = _get_layer_raw_tensors(
        state_dict, layer_idx
    )
    # Single-device (mla_tp=1) expects per-TP shapes; slice if state dict has full logical (2-TP) size
    q_b, o_proj, kv_b1, kv_b2 = _slice_attention_weights_for_mla_tp(q_b, o_proj, kv_b1, kv_b2, bdw.mla_tp)
    logger.debug("  load raw tensors: {:.3f}s", time.perf_counter() - t0)
    logger.debug("Converting attention fusion groups for layer {} (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms)", layer_idx)
    t0 = time.perf_counter()
    q_a_proj, q_b_proj, kv_a_proj = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(
        q_a, q_b, kv_a, move_to_device=move_to_device
    )
    kv_b1_proj, kv_b2_proj = bdw.get_tt_kv_b12_proj_weights(kv_b1, kv_b2, move_to_device=move_to_device)
    logger.debug("  convert q_ab_kv_a + kv_b12: {:.3f}s", time.perf_counter() - t0)

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
        logger.debug("  convert o_proj_gate_mm_norms (MoE): {:.3f}s", time.perf_counter() - t0)
        attn = AttentionWeights(
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
        gate = GateWeights(gate_mm=gate_mm_ot, gate_bias=gate_bias_tt)
        return attn, gate
    else:
        gate_mm_dummy = torch.zeros(7168, 256, dtype=torch.bfloat16, device=next(iter(state_dict.values())).device)
        o_norms = bdw.get_tt_o_proj_and_gate_mm_weights(
            o_proj, gate_mm_dummy, attn_norm, q_norm, kv_norm, ffn_norm, move_to_device=move_to_device
        )
        o_proj_ot, _gate_mm_ot, attn_norm_ot, q_norm_ot, kv_norm_ot, ffn_norm_ot = o_norms
        logger.debug("  convert o_proj_gate_mm_norms (dense): {:.3f}s", time.perf_counter() - t0)
        attn = AttentionWeights(
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
        return attn, None


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
    move_to_device: bool = False,
) -> RoutedExpertWeights:
    """Prepare routed expert weights for one layer (dense: single MLP; MoE: num_routed_experts experts stacked)."""
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
            gate_stacked, up_stacked, down_stacked, move_to_device=move_to_device
        )
        logger.info("  converted routed experts in {:.3f}s", time.perf_counter() - t0)
        return RoutedExpertWeights(
            routed_gate_proj=routed_gate_proj,
            routed_up_proj=routed_up_proj,
            routed_down_proj=routed_down_proj,
        )
    else:
        mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T.contiguous()
        mlp_up = state_dict[_key(layer_idx, "mlp.up_proj.weight")].T.contiguous()
        mlp_down = state_dict[_key(layer_idx, "mlp.down_proj.weight")].T.contiguous()
        gate_t, up_t, down_t = bdw.get_tt_mlp_routed_expert_weights(
            mlp_gate, mlp_up, mlp_down, move_to_device=move_to_device
        )
        return RoutedExpertWeights(
            routed_gate_proj=[gate_t],
            routed_up_proj=[up_t],
            routed_down_proj=[down_t],
        )


def prepare_decoder_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    is_moe: bool,
    move_to_device: bool = False,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> DeepSeekV3DecoderLayerWeights:
    """Prepare fused weights for a single decoder layer (dense or MoE)."""
    logger.info("Preparing {} layer {}...", "MoE" if is_moe else "dense", layer_idx)
    t0 = time.perf_counter()
    attn, gate = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=is_moe, move_to_device=move_to_device)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=is_moe, move_to_device=move_to_device)
    routed = prepare_routed_expert_weights(
        bdw,
        state_dict,
        layer_idx,
        is_moe=is_moe,
        num_routed_experts=num_routed_experts,
        move_to_device=move_to_device,
    )
    if is_moe:
        assert gate is not None
        result: DeepSeekV3DecoderLayerWeights = DeepSeekV3MoELayerWeights(
            attention=attn,
            gate=gate,
            shared_expert=shared,
            routed_expert=routed,
        )
    else:
        result = DeepSeekV3DenseLayerWeights(
            attention=attn,
            shared_expert=shared,
            routed_expert=routed,
        )
    logger.info("  layer {} done in {:.3f}s", layer_idx, time.perf_counter() - t0)
    return result


def prepare_dense_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    move_to_device: bool = False,
) -> DeepSeekV3DenseLayerWeights:
    """Prepare fused weights for a single dense decoder layer."""
    layer = prepare_decoder_layer_weights(bdw, state_dict, layer_idx, is_moe=False, move_to_device=move_to_device)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)
    return layer


def prepare_moe_layer_weights(
    bdw: BlitzDecodeWeights,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> DeepSeekV3MoELayerWeights:
    """Prepare fused weights for a single MoE decoder layer."""
    layer = prepare_decoder_layer_weights(
        bdw,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_routed_experts,
    )
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    return layer


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
    logger.info("Dump embedding weights...")
    ttnn.dump_tensor(emb_dir / "embedding.tensorbin", weights.embedding)
    logger.info("Dump manifest...")
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
    gate: GateWeights | None = None,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save attention fusion groups to layer dir; merge into existing manifest if present.
    When is_moe is True, gate must be provided and gate_mm/gate_bias are saved from it.
    """
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
    if gate is not None:
        field_tuples.append(("gate_mm", gate.gate_mm))
    new_groups = _dump_overlapped_fusion_groups(layer_dir, field_tuples)
    manifest.setdefault("fusion_groups", {}).update(new_groups)
    if gate is not None:
        ttnn.dump_tensor(layer_dir / "gate_bias.tensorbin", gate.gate_bias)
        manifest.setdefault("standalone_tensors", {})["gate_bias"] = "gate_bias.tensorbin"
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
    routed: RoutedExpertWeights,
    path: str | Path,
    layer_idx: int,
    *,
    is_moe: bool,
    hf_model_name: str = "",
    hf_state_dict_name: str = "",
    device_mesh_shape: tuple[int, int] = (1, 1),
) -> None:
    """Save routed expert weights to layer dir; merge into existing manifest if present.
    MoE: experts/e_NNN/{gate,up,down}_proj.tensorbin (one dir per expert, 3 files each).
    Dense: 1 file per projection in layer_dir (routed_gate_proj.tensorbin, etc.).
    """
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
    logger.debug("Saving routed expert weights for layer {}...", layer_idx)
    if is_moe:
        num_experts = len(routed.routed_gate_proj)
        manifest["routed_experts"] = {"num_experts": num_experts}
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
    else:
        ttnn.dump_tensor(layer_dir / "routed_gate_proj.tensorbin", routed.routed_gate_proj[0])
        ttnn.dump_tensor(layer_dir / "routed_up_proj.tensorbin", routed.routed_up_proj[0])
        ttnn.dump_tensor(layer_dir / "routed_down_proj.tensorbin", routed.routed_down_proj[0])
        standalone["routed_gate_proj"] = "routed_gate_proj.tensorbin"
        standalone["routed_up_proj"] = "routed_up_proj.tensorbin"
        standalone["routed_down_proj"] = "routed_down_proj.tensorbin"
    with open(layer_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def save_decoder_layer(
    layer: DeepSeekV3DecoderLayerWeights,
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
    save_decoder_layer_t0 = time.perf_counter()
    attn = layer.attention
    shared = layer.shared_expert
    routed = layer.routed_expert
    gate = layer.gate if is_moe else None
    save_attention_weights(
        attn,
        path,
        layer_idx,
        is_moe=is_moe,
        gate=gate,
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
    logger.info(f"  save_decoder_layer total: {time.perf_counter() - save_decoder_layer_t0:.3f}s")


def load_moe_routed_experts(
    path: str | Path,
    device,
    layer_idx: int,
) -> RoutedExpertWeights:
    """Load only the routed expert weights for an MoE layer from cache.

    Reads experts/e_NNN/{gate,up,down}_proj.tensorbin. Since setup_fast_dispatch can
    only be used once per program, call this under setup_fast_dispatch and pass the
    result to load_moe_decoder_layer(..., preloaded_routed_experts=...). If you do
    not use fast dispatch, omit preloaded_routed_experts and load_moe_decoder_layer
    will load from disk.
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
    num_experts = manifest.get("routed_experts", {}).get("num_experts", NUM_ROUTED_EXPERTS)
    experts_dir = layer_dir / "experts"
    logger.info("Loading routed experts for layer {} from cache...", layer_idx)
    t0 = time.perf_counter()
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
    logger.info("  routed experts for layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - t0)
    return RoutedExpertWeights(
        routed_gate_proj=routed_gate_proj,
        routed_up_proj=routed_up_proj,
        routed_down_proj=routed_down_proj,
    )


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
    routed_gate_proj = [ttnn.load_tensor(layer_dir / standalone["routed_gate_proj"], device=device)]
    routed_up_proj = [ttnn.load_tensor(layer_dir / standalone["routed_up_proj"], device=device)]
    routed_down_proj = [ttnn.load_tensor(layer_dir / standalone["routed_down_proj"], device=device)]
    logger.info("  layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - load_t0)

    attention = AttentionWeights(
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
    )
    shared_expert = SharedExpertWeights(
        shared_gate_proj=shared_gate_proj,
        shared_up_proj=shared_up_proj,
        shared_down_proj=shared_down_proj,
    )
    routed_expert = RoutedExpertWeights(
        routed_gate_proj=routed_gate_proj,
        routed_up_proj=routed_up_proj,
        routed_down_proj=routed_down_proj,
    )
    return DeepSeekV3DenseLayerWeights(
        attention=attention,
        shared_expert=shared_expert,
        routed_expert=routed_expert,
    )


def load_moe_decoder_layer(
    path: str | Path,
    device,
    layer_idx: int,
    *,
    preloaded_routed_experts: RoutedExpertWeights | None = None,
) -> DeepSeekV3MoELayerWeights:
    """Deserialize an MoE decoder layer from <path>/layer_{layer_idx:03d}/.

    If preloaded_routed_experts is provided (e.g. from load_moe_routed_experts under
    setup_fast_dispatch, which can only be used once per program), those weights are
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
    routed_expert = preloaded_routed_experts
    logger.info("  layer {} loaded in {:.3f}s", layer_idx, time.perf_counter() - load_t0)

    attention = AttentionWeights(
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
    )
    gate = GateWeights(gate_mm=gate_mm, gate_bias=gate_bias)
    shared_expert = SharedExpertWeights(
        shared_gate_proj=shared_gate_proj,
        shared_up_proj=shared_up_proj,
        shared_down_proj=shared_down_proj,
    )
    return DeepSeekV3MoELayerWeights(
        attention=attention,
        gate=gate,
        shared_expert=shared_expert,
        routed_expert=routed_expert,
    )
