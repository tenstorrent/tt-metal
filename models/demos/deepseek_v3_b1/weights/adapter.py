# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek **adapter** orchestration: map HF **SourceSelection** → preprocess → tensor-cache **artifacts** → **assemble** dataclasses."""

from __future__ import annotations

import time
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    GATE_UP_SPEC,
    KV_B12_SPEC,
    O_PROJ_GATE_MM_NORMS_SPEC,
    Q_AB_KV_A_SPEC,
    BlitzDecodeWeights,
    OverlappedTensor,
)
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, SourceTensorSelection
from models.demos.deepseek_v3_b1.weights import catalog as C
from models.demos.deepseek_v3_b1.weights.preprocessing import (
    deinterleave_q_b_proj,
    mtp_eh_proj_preprocess,
    split_kv_b_proj,
)
from models.demos.deepseek_v3_b1.weights.types import (
    _MTP_LAYER_IDX,
    _ROUTED_DOWN_K,
    _ROUTED_DOWN_N,
    _ROUTED_GATE_UP_K,
    _ROUTED_GATE_UP_N,
    NUM_ROUTED_EXPERTS,
    AttentionWeights,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    SharedExpertWeights,
)

# Per-TP attention tensor dimensions (match BlitzDecodeWeights configs for single device)
_MLA_TP1_Q_B_WIDTH = 12288
_MLA_TP1_O_PROJ_HEIGHT = 8192
_MLA_TP1_KV_B1_HEIGHT = 8192
_MLA_TP1_KV_B2_WIDTH = 8192

# Per-TP shared expert dimensions (gate/up (7168, 256), down (256, 7168) for moe_tp=1)
_MOE_TP1_SHARED_GATE_UP_N = 256
_MOE_TP1_SHARED_DOWN_K = 256

# Gate routing constants (bias/indices layout on sender core)
_GATE_BIAS_INDICES_SHAPE = (16, 16)
_GATE_NUM_INDICES = D.GATE_NUM_INDICES


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
    from models.demos.deepseek_v3_b1.weights.catalog import _GATE_BIAS_SENDER_CORE_GRID, _GATE_BIAS_TILE

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
        target = C.gate_bias_target(layer_idx)
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
    # Dense and MoE intentionally share the same fused layout target for o_proj+norms.
    # Dense uses a synthetic zero gate_mm in preprocess so the packed format remains identical.
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
        sd_target = C.shared_down_tensor_target(bdw)
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
        shared_n = D.MOE_INTERMEDIATE_SIZE

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
        sd_target = C.shared_down_tensor_target(bdw)
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
        tgt_gate = C.moe_routed_expert_tensor_target("routed_gate_proj", _ROUTED_GATE_UP_K, _ROUTED_GATE_UP_N, device)
        tgt_up = C.moe_routed_expert_tensor_target("routed_up_proj", _ROUTED_GATE_UP_K, _ROUTED_GATE_UP_N, device)
        tgt_down = C.moe_routed_expert_tensor_target("routed_down_proj", _ROUTED_DOWN_K, _ROUTED_DOWN_N, device)
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
        tgt_g = C.dense_routed_stacked_tensor_target("routed_gate_proj", _ROUTED_GATE_UP_K, _ROUTED_GATE_UP_N, device)
        tgt_u = C.dense_routed_stacked_tensor_target("routed_up_proj", _ROUTED_GATE_UP_K, _ROUTED_GATE_UP_N, device)
        tgt_d = C.dense_routed_stacked_tensor_target("routed_down_proj", _ROUTED_DOWN_K, _ROUTED_DOWN_N, device)

        fp_g = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(gate_k,)),
            target=tgt_g,
        )

        _dn_shared = D.MOE_INTERMEDIATE_SIZE
        _dn_num_routed = (D.INTERMEDIATE_SIZE - D.MOE_INTERMEDIATE_SIZE) // D.MOE_INTERMEDIATE_SIZE
        _dn_expert_n = D.MOE_INTERMEDIATE_SIZE

        def _pre_routed_gate(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            mg = t[gate_k].T.contiguous()
            ge = mg[:, _dn_shared:].reshape(mg.shape[0], _dn_num_routed, _dn_expert_n).permute(1, 0, 2).contiguous()
            return {"routed_gate_proj": bdw.mlp_routed_dense_stacked_torch_for_cache(ge)}

        fp_u = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(up_k,)),
            target=tgt_u,
        )

        def _pre_routed_up(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            mu = t[up_k].T.contiguous()
            ue = mu[:, _dn_shared:].reshape(mu.shape[0], _dn_num_routed, _dn_expert_n).permute(1, 0, 2).contiguous()
            return {"routed_up_proj": bdw.mlp_routed_dense_stacked_torch_for_cache(ue)}

        fp_d = cache_config.context.fingerprint(
            source=SourceTensorSelection(names=(down_k,)),
            target=tgt_d,
        )

        def _pre_routed_down(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            md = t[down_k].T.contiguous()
            de = md[_dn_shared:, :].reshape(_dn_num_routed, _dn_expert_n, md.shape[1]).contiguous()
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
        target=C.EMBEDDING_TARGET,
    )
    embedding_tt = cache_config.cache.get_or_create(
        fingerprint,
        device,
        preprocess=_preprocess_embedding,
        raw_tensors=lambda: {_src_key: state_dict[_src_key]},
    )
    return DeepSeekV3EmbeddingLayerWeights(embedding=embedding_tt)


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
            C._LM_HEAD_VOCAB_SIZE,
            C._LM_HEAD_K,
        ), f"Expected lm_head shape ({C._LM_HEAD_VOCAB_SIZE}, {C._LM_HEAD_K}), got {lm_w.shape}"
        return {"lm_head": lm_w.T.contiguous()}

    lm_fingerprint = cache_config.context.fingerprint(
        source=SourceTensorSelection(names=(_lm_key,)),
        target=C.LM_HEAD_TARGET,
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
        target=C.FINAL_NORM_TARGET,
    )
    final_norm_tt = cache_config.cache.get_or_create(
        norm_fingerprint,
        device,
        preprocess=_preprocess_final_norm,
        raw_tensors=lambda: {_norm_key: state_dict[_norm_key]},
    )

    return DeepSeekV3LMHeadWeights(lm_head=lm_head_tt, final_norm=final_norm_tt)


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
    h_target = C.mtp_norm_target("mtp_h_gamma")
    h_fingerprint = cache_config.context.fingerprint(source=SourceTensorSelection(names=(_h_key,)), target=h_target)
    h_gamma_tt = cache_config.cache.get_or_create(
        h_fingerprint,
        device,
        preprocess=lambda t: {h_target.name: t[_h_key].unsqueeze(0).contiguous()},
        raw_tensors=lambda: {_h_key: state_dict[_h_key]},
    )

    _e_key = _key(mtp_layer_idx, "enorm.weight")
    e_target = C.mtp_norm_target("mtp_e_gamma")
    e_fingerprint = cache_config.context.fingerprint(source=SourceTensorSelection(names=(_e_key,)), target=e_target)
    e_gamma_tt = cache_config.cache.get_or_create(
        e_fingerprint,
        device,
        preprocess=lambda t: {e_target.name: t[_e_key].unsqueeze(0).contiguous()},
        raw_tensors=lambda: {_e_key: state_dict[_e_key]},
    )

    _eh_key = _key(mtp_layer_idx, "eh_proj.weight")
    eh_target = C.mtp_eh_proj_target(K=2 * C._LM_HEAD_K, N=C._LM_HEAD_K)
    eh_fingerprint = cache_config.context.fingerprint(source=SourceTensorSelection(names=(_eh_key,)), target=eh_target)
    eh_proj_tt = cache_config.cache.get_or_create(
        eh_fingerprint,
        device,
        preprocess=lambda t: mtp_eh_proj_preprocess(t, _eh_key, eh_target.name),
        raw_tensors=lambda: {_eh_key: state_dict[_eh_key]},
    )

    logger.info("MTP weights prepared in {:.3f}s", time.perf_counter() - t0)
    return DeepSeekV3MTPWeights(
        h_gamma=h_gamma_tt,
        e_gamma=e_gamma_tt,
        eh_projection=eh_proj_tt,
    )
