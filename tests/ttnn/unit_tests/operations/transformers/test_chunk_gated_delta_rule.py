# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op correctness gate for ttnn.transformer.chunk_gated_delta_rule (F0 test pyramid, tier 3).

The public op is asserted against a torch golden of the WHOLE computation, inlined from
models/experimental/gated_attention_gated_deltanet/torch_functional/delta_rule_ops.py:149-245
(tests must not import models/), for BOTH dispatch paths: phased (QWEN_GDN_PHASED=1, the default
prep->scan split) and monolithic (QWEN_GDN_PHASED=0, single-kernel fallback; 4D inputs only —
the flat-QKV OPT-A inputs are phased-only).
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

CHUNK = 32  # Ct=1: the production chunk size
KDIM = 128
VDIM = 128


def _pcc(golden, actual):
    g = golden.to(torch.float64).flatten()
    a = actual.to(torch.float64).flatten()
    assert torch.isfinite(a).all(), "device output contains non-finite values"
    if torch.equal(g, a):
        return 1.0
    vg = g - g.mean()
    va = a - a.mean()
    denom = vg.norm() * va.norm()
    if denom == 0:
        return 0.0
    return float((vg @ va) / denom)


def _const_tiles(device, chunk_size=CHUNK):
    """The op's constant tiles (mirrors chunk_gated_delta_rule.cpp build_const_tiles)."""
    c = chunk_size
    eye = torch.eye(c, dtype=torch.float32)
    tril = torch.tril(torch.ones(c, c, dtype=torch.float32))
    ones = torch.ones(c, c, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    qtl = (lo_i & lo_j).float()
    qbr = (~lo_i & ~lo_j).float()
    qbl = (~lo_i & lo_j).float()
    masks = torch.cat([qtl, qbr, qbl], dim=1)  # [32, 96]

    def _up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return (_up(eye), _up(tril), _up(ones), _up(masks))


# ---------------------------------------------------------------------------
# Inlined torch golden (delta_rule_ops.py:149-245, use_qk_l2norm handled by the caller:
# the ttnn op requires host-normalized q/k, so the golden gets the already-normalized,
# bf16-rounded values the device sees)
# ---------------------------------------------------------------------------


def _golden_chunk_gdn(q, k, v, g, beta, scale, s0, C):
    """q,k: [B,T,H,K] fp32 (exact bf16-rounded, L2-normalized); v: [B,T,HV,V] fp32 (bf16-rounded);
    g,beta: [B,T,HV] fp32; s0: [B,HV,K,V] fp32 or None. T must be a multiple of C (no padding).
    Returns o [B,T,HV,V], final_state [B,HV,K,V]."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if G > 1:  # the op's GQA head expand (repeat_interleave over the head dim)
        q = q.repeat_interleave(G, dim=2)
        k = k.repeat_interleave(G, dim=2)
    # The op folds scale into q ON DEVICE in bf16 (ttnn::multiply on the bf16 tensor packs back
    # to bf16) — mirror that rounding so it doesn't count against the gates below.
    q = (q * scale).to(torch.bfloat16).to(torch.float32)

    BH, NC = B * HV, T // C
    q_c, k_c, v_c = (x.permute(0, 2, 1, 3).reshape(BH, NC, C, x.shape[-1]) for x in (q, k, v))
    g_c, beta_c = (x.permute(0, 2, 1).reshape(BH, NC, C) for x in (g, beta))

    decay = g_c.cumsum(-1)  # :189
    decay_exp = decay.exp().unsqueeze(-1)  # :190
    v_beta = v_c * beta_c.unsqueeze(-1)  # :171
    k_beta = k_c * beta_c.unsqueeze(-1)  # :172
    l_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()  # :193
    # :196-201 — WY inverse via forward substitution
    mask_upper = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k_c.transpose(-1, -2)) * l_mask).masked_fill(mask_upper, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    t_inv = attn + torch.eye(C, dtype=torch.float32)  # :201

    kd = k_beta * decay_exp
    q_decay = q_c * decay_exp  # :229
    mask_causal = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    intra = (q_c @ k_c.transpose(-1, -2) * l_mask).masked_fill(mask_causal, 0)  # :222
    k_dec_t = (k_c * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2)  # :237
    dl = decay[..., -1].exp()  # [BH,NC]

    # :216-238 scan loop in the phased op's un-premultiplied form — mathematically identical to
    # v_corrected/k_cumdecay premultiplication (v_new = t_inv @ (v_beta - kd@S) == u - w@S).
    S = s0.reshape(BH, K, V).clone() if s0 is not None else torch.zeros(BH, K, V, dtype=torch.float32)
    o = torch.zeros(BH, NC, C, V, dtype=torch.float32)
    for c in range(NC):
        v_new = t_inv[:, c] @ (v_beta[:, c] - kd[:, c] @ S)
        o[:, c] = q_decay[:, c] @ S + intra[:, c] @ v_new  # :229-232
        S = S * dl[:, c, None, None] + k_dec_t[:, c] @ v_new  # :235-238
    o = o.reshape(B, HV, T, V).permute(0, 2, 1, 3).contiguous()  # :243-244
    return o, S.reshape(B, HV, K, V)


# ---------------------------------------------------------------------------


# (1,256,8,8): MHA, BH=8. (1,512,4,8): GQA G=2 — exercises the repeat_interleave head expand.
@pytest.mark.parametrize("batch, seq, num_k_heads, num_v_heads", [(1, 256, 8, 8), (1, 512, 4, 8)])
@pytest.mark.parametrize("with_initial_state", [False, True])
@pytest.mark.parametrize("phased", [True, False], ids=["phased", "monolithic"])
def test_chunk_gated_delta_rule_vs_torch(
    device, monkeypatch, batch, seq, num_k_heads, num_v_heads, with_initial_state, phased
):
    torch.manual_seed(20260823)
    B, T, H, HV = batch, seq, num_k_heads, num_v_heads
    BH = B * HV
    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")

    monkeypatch.setenv("QWEN_GDN_PHASED", "1" if phased else "0")
    monkeypatch.delenv("QWEN_GDN_SCAN_SERIAL", raising=False)
    monkeypatch.delenv("QWEN_GDN_PREP_SERIAL", raising=False)
    # Mcast on/off is documented bit-exact, but pin the shipped default topology anyway.
    monkeypatch.delenv("QWEN_GDN_SCAN_MCAST", raising=False)
    # QWEN_GDN_DUMP is read once via a function-local static; delenv helps only if the op has not
    # run yet in this process — kept for hygiene.
    monkeypatch.delenv("QWEN_GDN_DUMP", raising=False)

    # The op's numeric regime: q/k L2-normalized on host (the op requires it — unnormalized keys
    # NaN the recurrence on every path), beta in (0,1), g <= 0, small initial state.
    q = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, HV, VDIM)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, HV))
    g = -F.softplus(torch.randn(B, T, HV)) * 0.5
    s0 = 0.05 * torch.randn(B, HV, KDIM, VDIM) if with_initial_state else None

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    eye, tril, ones, masks = _const_tiles(device)
    o_d, fs_d = ttnn.transformer.chunk_gated_delta_rule(
        dev(q, ttnn.bfloat16),
        dev(k, ttnn.bfloat16),
        dev(v, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
        initial_state=dev(s0, ttnn.float32) if s0 is not None else None,
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    o = ttnn.to_torch(o_d).float()  # phased o is fp32; monolithic o is bf16
    fs = ttnn.to_torch(fs_d).float()

    scale = KDIM**-0.5
    o_ref, fs_ref = _golden_chunk_gdn(q.float(), k.float(), v.float(), g, beta, scale, s0, CHUNK)

    # Gates: bf16 q/k/v inputs dominate the error on the phased path (kernel math is fp32/HiFi4
    # end-to-end, o packed fp32; fs accumulates only fp32 rounding, hence the tighter gate).
    # The monolithic path is looser on BOTH outputs: it uses the older numerics the phased path
    # abandoned for accuracy — a full-C Horner WY inverse (deep power series, more fp32
    # cancellation than the quadrant-split/forward-substitution forms) plus the premultiplied
    # u - w@S hand-off — and packs o to bf16.
    o_gate, fs_gate = (0.999, 0.9999) if phased else (0.998, 0.998)
    pcc_o = _pcc(o_ref, o)
    assert pcc_o >= o_gate, f"o: PCC {pcc_o} < {o_gate}"
    pcc_fs = _pcc(fs_ref, fs)
    assert pcc_fs >= fs_gate, f"final_state: PCC {pcc_fs} < {fs_gate}"
