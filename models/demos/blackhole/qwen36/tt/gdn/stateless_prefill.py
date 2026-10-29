# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip stateless GDN prefill: right-padded rows from a zero state, no state kept.

Rows are right-padded and the conv and the chunk op are causal, so a padded position never reaches a real one
and the final state is dropped: no valid_len masks are needed. The conv is the fused KDA op (conv + SiLU + q/k/v
split in one program, reading the TILE projection in place); q/k/v reach the chunk op flat, and o comes back
head-major (the same sequence as the tensor-parallel layer in tp.py).
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter
from models.demos.blackhole.qwen36.tt.prefill_matmul import prefill_linear


def stateless_prefill_supported(gdn, mode, stateless):
    """True when recurrent_forward may take this path: stateless chunk prefill on one device, fused chunk op,
    and no conv bias (the KDA conv has none)."""
    w = gdn.weights
    return (
        stateless
        and mode == "chunk"
        and gdn.device.get_num_devices() == 1
        and gdn.fused_prefill_fn is not None
        and w.mega_fused_weight is not None
        and w.fused_conv_weight_taps is not None
        and w.fused_conv_bias_dev is None
    )


def _consts(gdn):
    """uint32 [1] zero (the op's actual_start) and the zero conv history [1, K-1, C] ROW_MAJOR; built once per layer."""
    c = getattr(gdn, "_stateless_consts", None)
    if c is None:
        C = gdn.cfg.q_dim + gdn.cfg.k_dim + gdn.cfg.v_dim
        start = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=gdn.device,
        )
        hist = ttnn.from_torch(
            torch.zeros(1, gdn.conv_kernel_size - 1, C, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=gdn.device,
        )
        c = gdn._stateless_consts = (start, hist)
    return c


def stateless_prefill_forward(gdn, x):
    """x [B, S, dim] bf16 TILE DRAM, S a multiple of 32 -> [B, S, dim]. Weights and compute config as the
    mega-fused branch of gated_deltanet_forward_ttnn."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import _silu_mul, kda_channel_chunk_size

    w = gdn.weights
    B, S = x.shape[0], x.shape[1]
    Nk, Dk, Nv, Dv = gdn.num_heads, gdn.head_k_dim, gdn.num_v_heads, gdn.head_v_dim
    q_dim, k_dim, v_dim = gdn.cfg.q_dim, gdn.cfg.k_dim, gdn.cfg.v_dim
    C = q_dim + k_dim + v_dim
    assert S % 32 == 0, f"stateless GDN prefill: S={S} is not a multiple of 32"
    _dram = ttnn.DRAM_MEMORY_CONFIG

    # In-projection: qkv | a | b | z.
    mega = prefill_linear(x, w.mega_fused_weight, gdn.compute_kernel_config)
    a_off, b_off, g_off = w.mega_qkv_dim, w.mega_qkv_dim + w.mega_a_dim, w.mega_qkv_dim + w.mega_a_dim + w.mega_b_dim
    qkv = ttnn.to_layout(mega[:, :, :a_off], ttnn.TILE_LAYOUT)
    if a_off % 32 == 0 and g_off % 32 == 0:
        # Cut the a|b block on tile edges first; the unaligned a/b split then runs on [B, S, 32], not on mega.
        ab = ttnn.to_layout(mega[:, :, a_off:g_off], ttnn.TILE_LAYOUT)
        a = ttnn.to_layout(ab[:, :, : w.mega_a_dim], ttnn.TILE_LAYOUT)
        b = ttnn.to_layout(ab[:, :, w.mega_a_dim :], ttnn.TILE_LAYOUT)
        ttnn.deallocate(ab)
    else:
        a = ttnn.to_layout(mega[:, :, a_off:b_off], ttnn.TILE_LAYOUT)
        b = ttnn.to_layout(mega[:, :, b_off:g_off], ttnn.TILE_LAYOUT)
    z = ttnn.to_layout(mega[:, :, g_off:], ttnn.TILE_LAYOUT)
    ttnn.deallocate(mega)

    # Conv + SiLU + q/k/v split from a zero history. The op takes one [1, S, C] row: batch rows never mix.
    start, hist = _consts(gdn)
    cfg = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C))
    qs, ks, vs = [], [], []
    for r in range(B):
        row = qkv if B == 1 else ttnn.slice(qkv, (r, 0, 0), (r + 1, S, C), memory_config=_dram)
        qr, kr, vr = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            row,
            hist,
            *w.fused_conv_weight_taps,
            q_dim,
            k_dim,
            v_dim,
            program_config=cfg,
            actual_start=start,
            predecessor_carry=hist,
            memory_config=_dram,
        )
        if row is not qkv:
            ttnn.deallocate(row)
        qs.append(qr)
        ks.append(kr)
        vs.append(vr)
    ttnn.deallocate(qkv)
    if B == 1:
        q, k, v = qs[0], ks[0], vs[0]
    else:
        q, k, v = (ttnn.concat(t, dim=0, memory_config=_dram) for t in (qs, ks, vs))
        for t in qs + ks + vs:
            ttnn.deallocate(t)

    # Gates (fp32, the chunk op's contract; the values are the bf16 ones the masked path multiplied up to fp32).
    beta = ttnn.typecast(ttnn.sigmoid(b), ttnn.float32)
    ttnn.deallocate(b)
    g = ttnn.typecast(ttnn.multiply(w.A_neg, ttnn.softplus(ttnn.add(a, w.dt_bias))), ttnn.float32)
    ttnn.deallocate(a)
    beta = ttnn.reshape(beta, (B, S, Nv))
    g = ttnn.reshape(g, (B, S, Nv))

    # Chunk op: flat q/k/v in, head-major o out; the final state is dropped.
    o, final_state = chunk_gated_delta_rule_fused_adapter(
        q,
        k,
        v,
        beta,
        g,
        scale=Dk**-0.5,
        initial_state=None,
        device=gdn.device,
        qkv_head_dims=(Nk, Dk, Nv, Dv),
        return_o_bh=True,
        const_tiles=gdn.fused_prefill_fn.keywords["const_tiles"],
    )
    ttnn.deallocate(final_state)
    for t in (q, k, v, beta, g):
        ttnn.deallocate(t)

    # Per-head norm, head -> token relayout, gate (bf16 out), out-projection.
    n = ttnn.rms_norm(o, weight=w.o_norm_weight, epsilon=gdn.norm_eps)
    ttnn.deallocate(o)
    n = ttnn.reshape(n, (B, Nv, S, Dv))
    n = ttnn.experimental.nlp_concat_heads(n)
    n = ttnn.reshape(n, (B, S, Nv * Dv))
    gated = _silu_mul(n, z, _dram, dtype=ttnn.bfloat16)
    ttnn.deallocate(n)
    ttnn.deallocate(z)
    out = prefill_linear(gated, w.o_proj_weight, gdn.compute_kernel_config)
    ttnn.deallocate(gated)
    return out
