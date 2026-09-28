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
    """TtFullAttention / TtSlidingAttention (TP=4 over the 2x2 mesh) for one layer: fused qkv dequantized per TP rank,
    bf16 o_proj; sliding layers add the window and the per-head sink (chip d holds sink values 16d..16d+15)."""
    import torch

    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rope_inv_freq
    from models.demos.mimo_v2_6_d_p.reference.weights import qkv_weight
    from models.demos.mimo_v2_6_d_p_2x2.tt.attention import TtFullAttention, TtSlidingAttention

    hq, hkv, d, dv = cfg.attn_dims(layer)
    p = f"model.layers.{layer}.self_attn."
    wqkv = qkv_weight(loader, p, (hq * d, hkv * d, hkv * dv), torch.float32)
    wo = loader.get(p + "o_proj.weight").float()
    sliding = cfg.is_sliding(layer)
    inv_freq = rope_inv_freq(cfg.swa_rope_theta if sliding else cfg.rope_theta, cfg.rope_dim(layer))
    dims, vs = (hq, hkv, d, dv), cfg.attention_value_scale
    if sliding:
        sink = loader.get(p + "attention_sink_bias").float() if cfg.has_sink(layer) else None
        return TtSlidingAttention(mesh, wqkv, wo, dims, inv_freq, max_seq, vs, cfg.sliding_window, sink)
    assert not cfg.has_sink(layer)
    return TtFullAttention(mesh, wqkv, wo, dims, inv_freq, max_seq, vs)


def new_kv_cache(mesh, cfg, layer: int, max_seq: int):
    """Empty device KV cache for one layer (K 192 wide, V 128): full layers 4 KV heads (head d on chip d = 2*row + col,
    paged-shaped, identity page table resident), sliding layers 8 KV heads (heads 2d, 2d+1 on chip d, contiguous)."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.attention import TtKVCacheFull, TtKVCacheSliding

    _, hkv, d, dv = cfg.attn_dims(layer)
    if cfg.is_sliding(layer):
        return TtKVCacheSliding(mesh, hkv, d, dv, max_seq)
    return TtKVCacheFull(mesh, hkv, d, dv, max_seq)


def build_mlp(mesh, loader, layer: int):
    """TtDenseMLP (TP=4 SwiGLU over the 2x2 mesh, all_reduce over both axes); fp8 + 128x128 block scale dequantized
    to bf16 at load."""
    import torch

    from models.demos.mimo_v2_6_d_p.reference.weights import fp8_weight
    from models.demos.mimo_v2_6_d_p_2x2.tt.mlp import TtDenseMLP

    p = f"model.layers.{layer}.mlp."
    wg, wu, wd = (fp8_weight(loader, p + f"{n}.weight", torch.float32) for n in ("gate_proj", "up_proj", "down_proj"))
    return TtDenseMLP(mesh, wg, wu, wd)
