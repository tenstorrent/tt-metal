# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash-RL text decoder on the 2x2 mesh: module builders shared by the component hooks and (later) the
all-device model, so both paths construct identical modules. Grows one validated component at a time."""

from __future__ import annotations

# Norm steps -> checkpoint weight name (under model.layers.<i>.).
NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def build_norm(mesh, loader, layer: int, step: str, eps: float = 1e-6):
    """TtRMSNorm (replicated on all 4 chips, HiFi4 + fp32 acc, plain w) for one layer's norm step."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import TtRMSNorm

    return TtRMSNorm(mesh, loader.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}"), eps=eps)


def build_attention(mesh, loader, cfg, layer: int, max_seq: int):
    """TtFullAttention (TP=4 over the 2x2 mesh) for one full-attention layer: fused qkv dequantized per TP rank, bf16
    o_proj, no sink. Sliding layers are not ported yet."""
    import torch

    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rope_inv_freq
    from models.demos.mimo_v2_6_d_p.reference.weights import qkv_weight
    from models.demos.mimo_v2_6_d_p_2x2.tt.attention import TtFullAttention

    if cfg.is_sliding(layer):
        raise NotImplementedError(f"layer {layer}: sliding attention not ported to 2x2 yet")
    assert not cfg.has_sink(layer)
    hq, hkv, d, dv = cfg.attn_dims(layer)
    p = f"model.layers.{layer}.self_attn."
    wqkv = qkv_weight(loader, p, (hq * d, hkv * d, hkv * dv), torch.float32)
    wo = loader.get(p + "o_proj.weight").float()
    inv_freq = rope_inv_freq(cfg.rope_theta, cfg.rope_dim(layer))
    return TtFullAttention(mesh, wqkv, wo, (hq, hkv, d, dv), inv_freq, max_seq, cfg.attention_value_scale)


def new_kv_cache(mesh, cfg, layer: int, max_seq: int):
    """Empty device KV cache for one full-attention layer (K 192 wide, V 128; KV head d on chip d = 2*row + col,
    paged-shaped, identity page table resident)."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.attention import TtKVCacheFull

    if cfg.is_sliding(layer):
        raise NotImplementedError(f"layer {layer}: sliding KV cache not ported to 2x2 yet")
    _, hkv, d, dv = cfg.attn_dims(layer)
    return TtKVCacheFull(mesh, hkv, d, dv, max_seq)
