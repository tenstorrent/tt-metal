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
