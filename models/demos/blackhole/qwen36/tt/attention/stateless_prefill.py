# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip stateless gated-attention prefill: plain causal SDPA over [B, H, S, D], no KV cache, no state kept.

Two projections produce q | k | v and the gate; ``nlp_create_qkv_heads`` splits the heads straight into [B, H, S, D],
so the reshape, chunk and transpose ops of the generic path (gated_attention_forward_ttnn) are gone. Norms, RoPE, SDPA,
gate and out-projection are the generic path's own helpers and configs.
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.prefill_matmul import prefill_linear
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _get_sdpa_compute_kernel_config,
    _get_sdpa_program_config,
)


def _sdpa_program_config(device, B, H, S):
    """Shared SDPA config, but 128-wide q/k chunks (256-wide q at 1-2 waves, S >= 2048) for S < 8192 (S % 128 == 0) when B * H * S/128 fills the grid; approximate exp for S >= 2048."""
    cfg = _get_sdpa_program_config(device, S, q_seq_len=S)
    approx = S >= 2048
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    # 256-wide q chunks when B * H * S/256 is just above the core count (1-2 waves): (1,4096) 1465 -> 1019 us, (2,2048)
    # 799 -> 578 us; slower at 2-4 waves and for S < 2048 ((32,128) +25%, (16,256) +18%, (8,512) +3%, (4,1024) +2%).
    q_wide = S >= 2048 and S % 256 == 0 and cores <= B * H * S // 256 < 2 * cores
    if S < 8192 and S % 128 == 0 and B * H * (S // 128) >= cores:
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=256 if q_wide else 128,
            k_chunk_size=128,
            exp_approx_mode=approx,
        )
    if approx:
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
            q_chunk_size=cfg.q_chunk_size,
            k_chunk_size=cfg.k_chunk_size,
            exp_approx_mode=True,
        )
    return cfg


def _split_weights(attn):
    """(qkv [dim, (Hq + 2 Hkv) * D] = q heads | k heads | v heads, gate [dim, Hq * D]), bf8, built once per layer.

    q_proj is [dim, Hq * 2D] with (q, gate) interleaved per head; whole D-column head blocks are tile aligned, so the
    bfp8 round trip through host is exact."""
    qkv_w = getattr(attn, "_stateless_qkv_weight", None)
    gate_w = getattr(attn, "_stateless_gate_weight", None)
    if qkv_w is None or gate_w is None:
        w, cfg = attn.weights, attn.config
        Hq, D = cfg.num_heads, cfg.head_dim
        q_t = ttnn.to_torch(w.q_proj)
        dim = q_t.shape[0]
        qg = q_t.reshape(dim, Hq, 2, D)
        qkv = torch.cat(
            [qg[:, :, 0].reshape(dim, Hq * D), ttnn.to_torch(w.k_proj), ttnn.to_torch(w.v_proj)], dim=-1
        ).contiguous()
        gate = qg[:, :, 1].reshape(dim, Hq * D).contiguous()
        kw = dict(
            dtype=w.q_proj.dtype, layout=ttnn.TILE_LAYOUT, device=attn.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        qkv_w = attn._stateless_qkv_weight = ttnn.from_torch(qkv, **kw)
        gate_w = attn._stateless_gate_weight = ttnn.from_torch(gate, **kw)
    return qkv_w, gate_w


def _rope(x, cos, sin, trans_mat):
    """Partial RoPE of x [B, H, S, D] with one fused op: rotates the first cos.shape[-1] dims, the tail passes through.

    The q/k head dims must be stored rope-pair interleaved (pair (2j, 2j + 1)) and cos/sin [1, 1, S, rd] in the same
    layout; trans_mat is the 32x32 rotation matrix."""
    B, H, S, D = x.shape
    out = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
        ttnn.reshape(x, (1, B * H, S, D)), cos, sin, trans_mat, 0, 0, rotary_dim=cos.shape[-1], rotary_offset=0
    )
    return ttnn.reshape(out, (B, H, S, D))


def stateless_attention_prefill(attn, x, cos, sin, trans_mat):
    """x [B, S, dim] bf16 TILE DRAM, S a multiple of 32 -> [B, S, dim]. Same weights, norms, RoPE, SDPA configs and
    gate as the generic prefill (Qwen36GatedAttention.forward, no page table), without the KV cache. The q/k weights
    must be rope-pair permuted (see _rope); cos/sin [1, 1, S, rd] in the pair layout."""
    w, cfg = attn.weights, attn.config
    B, S = x.shape[0], x.shape[1]
    Hq, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    assert S % 32 == 0, f"stateless attention prefill: S={S} is not a multiple of 32"
    ckc = attn.compute_kernel_config
    qkv_w = (Hq + 2 * Hkv) * D

    # Projection: two matmuls (q | k | v, gate), so no output slices.
    qkv_weight, gate_weight = _split_weights(attn)
    qkv = ttnn.reshape(prefill_linear(x, qkv_weight, ckc), [B, 1, S, qkv_w])
    gate = prefill_linear(x, gate_weight, ckc)

    # Heads straight to [B, H, S, D].
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=Hq, num_kv_heads=Hkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(qkv)

    # Per-head q/k norm (weights carry the +1 offset), then fused partial RoPE on the pair-interleaved dims.
    q = ttnn.rms_norm(q, weight=w.q_norm, epsilon=cfg.norm_eps)
    k = ttnn.rms_norm(k, weight=w.k_norm, epsilon=cfg.norm_eps)
    q_n, k_n = q, k
    q, k = _rope(q_n, cos, sin, trans_mat), _rope(k_n, cos, sin, trans_mat)
    ttnn.deallocate(q_n)
    ttnn.deallocate(k_n)

    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        scale=D**-0.5,
        program_config=_sdpa_program_config(attn.device, B, Hq, S),
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
    res = prefill_linear(gated, w.o_proj, ckc)
    ttnn.deallocate(gated)
    return res
