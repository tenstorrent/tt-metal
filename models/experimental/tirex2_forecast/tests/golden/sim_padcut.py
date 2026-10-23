# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU only: evidence for candidate (b). Start every layer from states cached after the 192 all-NaN
left-pad tokens (computed once on one all-pad row) and run only tokens 192..284; compare with upstream.

    python models/experimental/tirex2_forecast/tests/golden/sim_padcut.py
"""

import torch, torch.nn.functional as F
from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden

torch.set_num_threads(16)
W = df.build_weights(df.load_state_dict(df.default_checkpoint()))
g = golden("model16")
prep = df.prepare(g["context"], 64)
P = 192
assert bool((prep.tokens[:, :P] == 0).all())


def mlstm_from_state(xn, w, state, chunk):
    B, S, _ = xn.shape
    p = df.linear(xn, w.w_qkvoif)
    q, k, v, o, gg = torch.split(p, [256, 256, 512, 512, 8], dim=-1)
    gg = df.soft_cap(gg + w.b_if)
    th = lambda t: t.reshape(B, S, 4, -1).transpose(1, 2)
    i, f = gg[..., :4].transpose(1, 2), gg[..., 4:].transpose(1, 2)
    # chunkwise with an initial state: reuse dataflow by prepending nothing; implement inline
    C0, n0, m0 = state
    q, k, v = th(q), th(k), th(v)
    pad = (-S) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        i = F.pad(i, (0, pad), value=-25.0)
        f = F.pad(f, (0, pad), value=25.0)
    nc = (S + pad) // chunk
    L = chunk
    qc = q.reshape(B, 4, nc, L, 64) * 0.125
    kc = k.reshape(B, 4, nc, L, 64)
    vc = v.reshape(B, 4, nc, L, 128)
    ic = i.reshape(B, 4, nc, L)
    bc = F.logsigmoid(f).reshape(B, 4, nc, L).cumsum(-1)
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    C, n, m = C0.expand(B, 4, 64, 128), n0.expand(B, 4, 64), m0.expand(B, 4)
    out = []
    for c in range(nc):
        b_, i_, q_, k_, v_ = bc[:, :, c], ic[:, :, c], qc[:, :, c], kc[:, :, c], vc[:, :, c]
        D = (b_[..., :, None] - b_[..., None, :] + i_[..., None, :]).masked_fill(~causal, float("-inf"))
        m_t = torch.maximum(b_ + m[..., None], D.max(-1).values)
        Wt = torch.exp(D - m_t[..., None]) * (q_ @ k_.transpose(-1, -2))
        inter = torch.exp(b_ + m[..., None] - m_t)[..., None]
        num = inter * (q_ @ C) + Wt @ v_
        den = inter[..., 0] * (q_ @ n[..., None])[..., 0] + Wt.sum(-1)
        out.append(num / (torch.maximum(den.abs(), torch.exp(-m_t)) + 1e-6)[..., None])
        gl = b_[..., -1]
        a = gl[..., None] - b_ + i_
        mn = torch.maximum(gl + m, a.max(-1).values)
        ka = k_ * torch.exp(a - mn[..., None])[..., None]
        dec = torch.exp(gl + m - mn)
        C = dec[..., None, None] * C + ka.transpose(-1, -2) @ v_
        n = dec[..., None] * n + ka.sum(-2)
        m = mn
    h = torch.cat(out, dim=2)[:, :, :S].transpose(1, 2).reshape(B, S, -1)
    return df.linear(torch.sigmoid(o) * df.mh_layer_norm(h, w.w_mhnorm), w.w_out)


# 1. per-layer cached states from one all-pad row (fp32 CPU, once per P)
x = df.residual_block(torch.zeros(1, P, 64), W.embed)
states = []
for blk in W.blocks:
    xn = df.rms_norm(x, blk.w_norm_lstm)
    if df.is_mlstm(blk.index):
        cap = {}
        y = df.mlstm_cell(xn, blk.cell, capture=cap)
        _, st = df.mlstm_chunkwise(cap["q"], cap["k"], cap["v"], cap["i"], cap["f"], chunk=64, return_states=True)
        states.append(st[-1] if P % 64 == 0 else None)
    else:
        Wx = df.slstm_input(xn, blk.cell)
        yy, sts = df.slstm_recurrence(Wx, blk.cell.r)
        y = df.mh_layer_norm(yy, blk.cell.w_gnorm)
        states.append(sts[-1])
    x = x + y
    x = x + df.ffn(df.rms_norm(x, blk.w_norm_ffn), blk.ffn)
# 2. the short pass over tokens [P, 285)
for chunk in (64, 32, 96):
    x = df.residual_block(prep.tokens[:, P:], W.embed)
    for blk, st in zip(W.blocks, states):
        xn = df.rms_norm(x, blk.w_norm_lstm)
        if df.is_mlstm(blk.index):
            y = mlstm_from_state(xn, blk.cell, st, chunk)
        else:
            Wx = df.slstm_input(xn, blk.cell)
            B = xn.shape[0]
            yy, _ = df.slstm_recurrence(Wx, blk.cell.r, state=tuple(s.expand(B, 512) for s in st))
            y = df.mh_layer_norm(yy, blk.cell.w_gnorm)
        x = x + y
        x = x + df.ffn(df.rms_norm(x, blk.w_norm_ffn), blk.ffn)
    head = df.residual_block(df.t5_norm(x, W.w_out_norm), W.head)
    tok = [t - P for t in df.readout_tokens(64)]
    ref = g["head_out"][:, df.readout_tokens(64)]
    got = head[:, tok]
    print(
        f"[padcut P={P} chunk={chunk}] head readout rel err vs golden {float((got - ref).abs().max() / ref.abs().max()):.2e}; block11 rel err {float((x - g['block11.out'][:, P:]).abs().max() / g['block11.out'][:, P:].abs().max()):.2e}"
    )
