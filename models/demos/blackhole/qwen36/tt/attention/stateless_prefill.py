# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip stateless gated-attention prefill: plain causal SDPA over [B, H, S, D], no KV cache, no state kept.

One projection produces q | k | v | gate; ``nlp_create_qkv_heads`` splits the heads straight into [B, H, S, D], so the
reshape, chunk and transpose ops of the generic path (gated_attention_forward_ttnn) are gone. Norms, RoPE, SDPA,
gate and out-projection are the generic path's own helpers and configs.
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.prefill_matmul import prefill_linear
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _get_sdpa_compute_kernel_config,
    _get_sdpa_program_config,
)


def _fused_weight(attn):
    """[dim, (Hq + 2 Hkv + Hq) * D] = q heads | k heads | v heads | gate heads, bf8, built once per layer.

    q_proj is [dim, Hq * 2D] with (q, gate) interleaved per head; whole D-column head blocks are tile aligned, so the
    bfp8 round trip through host is exact."""
    fw = getattr(attn, "_stateless_qkvg_weight", None)
    if fw is None:
        w, cfg = attn.weights, attn.config
        Hq, D = cfg.num_heads, cfg.head_dim
        q_t = ttnn.to_torch(w.q_proj)
        dim = q_t.shape[0]
        qg = q_t.reshape(dim, Hq, 2, D)
        fused = torch.cat(
            [
                qg[:, :, 0].reshape(dim, Hq * D),
                ttnn.to_torch(w.k_proj),
                ttnn.to_torch(w.v_proj),
                qg[:, :, 1].reshape(dim, Hq * D),
            ],
            dim=-1,
        ).contiguous()
        fw = attn._stateless_qkvg_weight = ttnn.from_torch(
            fused,
            dtype=w.q_proj.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=attn.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return fw


def _partial_rope(x, cos, sin):
    """HF rotate-half RoPE on the first cos.shape[-1] dims of x [B, H, S, D] (fused op); the tail passes through.
    cos/sin [1, S, rd]."""
    B, H, S, D = x.shape
    rd = cos.shape[-1]
    cos = ttnn.reshape(cos, (1, 1, S, rd))
    sin = ttnn.reshape(sin, (1, 1, S, rd))
    x_rot = ttnn.reshape(ttnn.slice(x, (0, 0, 0, 0), (B, H, S, rd)), (1, B * H, S, rd))
    roped = ttnn.experimental.rotary_embedding_hf(x_rot, cos, sin, is_decode_mode=False)
    ttnn.deallocate(x_rot)
    roped = ttnn.reshape(roped, (B, H, S, rd))
    if rd == D:
        return roped
    x_pass = ttnn.slice(x, (0, 0, 0, rd), (B, H, S, D))
    res = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return res


def stateless_attention_prefill(attn, x, cos, sin):
    """x [B, S, dim] bf16 TILE DRAM, S a multiple of 32 -> [B, S, dim]. Same weights, norms, RoPE, SDPA configs and
    gate as the generic prefill (Qwen36GatedAttention.forward, no page table), without the KV cache."""
    w, cfg = attn.weights, attn.config
    B, S = x.shape[0], x.shape[1]
    Hq, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    assert S % 32 == 0, f"stateless attention prefill: S={S} is not a multiple of 32"
    ckc = attn.compute_kernel_config
    qkv_w = (Hq + 2 * Hkv) * D

    # Projection: q | k | v | gate.
    out = prefill_linear(x, _fused_weight(attn), ckc)
    qkv = ttnn.reshape(out[:, :, :qkv_w], [B, 1, S, qkv_w])
    gate = out[:, :, qkv_w:]

    # Heads straight to [B, H, S, D].
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=Hq, num_kv_heads=Hkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(qkv)

    # Per-head q/k norm (weights carry the +1 offset), partial RoPE.
    q = ttnn.rms_norm(q, weight=w.q_norm, epsilon=cfg.norm_eps)
    k = ttnn.rms_norm(k, weight=w.k_norm, epsilon=cfg.norm_eps)
    q_n, k_n = q, k
    q, k = _partial_rope(q_n, cos, sin), _partial_rope(k_n, cos, sin)
    ttnn.deallocate(q_n)
    ttnn.deallocate(k_n)

    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        scale=D**-0.5,
        program_config=_get_sdpa_program_config(attn.device, S, q_seq_len=S),
        compute_kernel_config=_get_sdpa_compute_kernel_config(),
    )
    for t in (q, k, v):
        ttnn.deallocate(t)

    # Heads -> [B, S, Hq * D], sigmoid gate, out-projection.
    attn_out = ttnn.experimental.nlp_concat_heads(attn_out)
    attn_out = ttnn.reshape(attn_out, [B, S, Hq * D])
    gated = ttnn.multiply(attn_out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(gate)
    ttnn.deallocate(attn_out)
    ttnn.deallocate(out)
    res = prefill_linear(gated, w.o_proj, ckc)
    ttnn.deallocate(gated)
    return res
