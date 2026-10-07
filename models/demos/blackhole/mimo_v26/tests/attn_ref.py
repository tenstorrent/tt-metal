# SPDX-License-Identifier: Apache-2.0
"""Torch fp32 reference for MiMo attention with HF semantics (modeling_mimo_v2.py): split-half partial RoPE on 64 of 192 dims, V*0.707 before cache,
scale 192**-0.5, GQA h -> h//(64/nkv), SWA window 128 incl self, SWA sink = raw logit column after scaling."""
import torch

NQ, DQK, DV, ROT = 64, 192, 128, 64


def rope_cos_sin(positions, theta):
    inv = theta ** (-torch.arange(0, ROT, 2).float() / ROT)
    f = positions.float()[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    return torch.cat([-x[..., ROT // 2 :], x[..., : ROT // 2]], -1)


def apply_rope(x, cos, sin):  # x [..., T, 192]
    xr, xp = x[..., :ROT], x[..., ROT:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], -1)


def ref_attention(x, q_w, k_w, v_w, o_w, sink, nkv, theta, window, v_scale=0.707):
    """x [T,4096] -> [T,4096], causal over the whole sequence, positions 0..T-1."""
    T = x.shape[0]
    x = x.float()
    q = (x @ q_w.float().T).view(T, NQ, DQK).transpose(0, 1)  # [64,T,192]
    k = (x @ k_w.float().T).view(T, nkv, DQK).transpose(0, 1)
    v = (x @ v_w.float().T).view(T, nkv, DV).transpose(0, 1) * v_scale
    cos, sin = rope_cos_sin(torch.arange(T), theta)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    qi = torch.arange(T)[:, None]
    kj = torch.arange(T)[None, :]
    mask = kj <= qi
    if window:
        mask = mask & (kj > qi - window)
    out = torch.zeros(NQ, T, DV)
    rep = NQ // nkv
    for h in range(NQ):
        s = (q[h] @ k[h // rep].T) * DQK**-0.5
        s = s.masked_fill(~mask, float("-inf"))
        if sink is not None:
            s = torch.cat([s, sink[h].float().reshape(1, 1).expand(T, 1)], 1)
        w = torch.softmax(s, 1)
        if sink is not None:
            w = w[:, :-1]
        out[h] = w @ v[h // rep]
    return out.transpose(0, 1).reshape(T, NQ * DV) @ o_w.float().T


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def ref_attention_rows(x, rows, q_w, k_w, v_w, o_w, sink, nkv, theta, window, v_scale=0.707, group=32):
    """Reference for selected query positions only (long contexts). x [T,4096] (positions 0..T-1); rows: sorted list of query positions; returns [len(rows),4096]."""
    rows = torch.as_tensor(rows)
    kpos = (
        torch.arange(int(rows.max()) + 1)
        if not window
        else torch.unique(torch.cat([torch.arange(max(0, int(r) - window + 1), int(r) + 1) for r in rows]))
    )
    K = torch.zeros(nkv, len(kpos), DQK)
    V = torch.zeros(nkv, len(kpos), DV)
    for s in range(0, len(kpos), 8192):
        idx = kpos[s : s + 8192]
        xs = x[idx].float()
        k = (xs @ k_w.float().T).view(len(idx), nkv, DQK).transpose(0, 1)
        v = (xs @ v_w.float().T).view(len(idx), nkv, DV).transpose(0, 1) * v_scale
        c, sn = rope_cos_sin(idx, theta)
        K[:, s : s + len(idx)] = apply_rope(k, c, sn)
        V[:, s : s + len(idx)] = v
    out = torch.zeros(len(rows), NQ * DV)
    rep = NQ // nkv
    for g0 in range(0, len(rows), group):
        rr = rows[g0 : g0 + group]
        q = (x[rr].float() @ q_w.float().T).view(len(rr), NQ, DQK).transpose(0, 1)  # [64,n,192]
        c, sn = rope_cos_sin(rr, theta)
        q = apply_rope(q, c, sn)
        mask = kpos[None, :] <= rr[:, None]
        if window:
            mask &= kpos[None, :] > rr[:, None] - window
        for h in range(NQ):
            s = (q[h] @ K[h // rep].T) * DQK**-0.5
            s = s.masked_fill(~mask, float("-inf"))
            if sink is not None:
                s = torch.cat([s, sink[h].float().reshape(1, 1).expand(len(rr), 1)], 1)
            w = torch.softmax(s, 1)
            if sink is not None:
                w = w[:, :-1]
            out[g0 : g0 + len(rr), h * DV : (h + 1) * DV] = w @ V[h // rep]
    return out @ o_w.float().T
