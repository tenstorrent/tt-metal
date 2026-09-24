# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1/M1 (host only): the standalone CPU reference against an inline torch golden, random weights.

Pattern: minimax_m3/tests/unit/test_reference_model.py. The inline golden below is written
independently of ``reference/qwen3_8_ref.py`` (token-by-token delta rule, explicit conv loop,
per-head attention loop) so the two oracles cannot drift apart silently.
"""

import pytest
import torch
import torch.nn.functional as F

from models.demos.qwen_3_8_27b.config import QWEN38
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref

SMALL = QWEN38.reduced(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=8,
    vocab_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    mrope_section=(3, 3, 2),
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------------- inline golden (fp32)
def g_rms(x, w, eps, unit_offset=True):
    x = x.float()
    y = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * ((1 + w.float()) if unit_offset else w.float())


def g_gdn(m, x, cfg):
    B, T, _ = x.shape
    qkv = x @ m.in_proj_qkv.weight.T  # [B,T,C]
    w = m.conv1d.weight[:, 0, :]  # [C,K]
    K = w.shape[1]
    pad = torch.cat([torch.zeros(B, K - 1, qkv.shape[-1]), qkv], 1)
    conv = sum(pad[:, j : j + T, :] * w[:, j] for j in range(K))
    conv = F.silu(conv)
    kd, vd = cfg.linear_key_dim, cfg.linear_value_dim
    q = conv[..., :kd].reshape(B, T, cfg.linear_num_key_heads, -1)
    k = conv[..., kd : 2 * kd].reshape(B, T, cfg.linear_num_key_heads, -1)
    v = conv[..., 2 * kd :].reshape(B, T, cfg.linear_num_value_heads, -1)
    rep = cfg.linear_num_value_heads // cfg.linear_num_key_heads
    q = F.normalize(q, dim=-1, eps=1e-6).repeat_interleave(rep, 2) / (q.shape[-1] ** 0.5)
    k = F.normalize(k, dim=-1, eps=1e-6).repeat_interleave(rep, 2)
    beta = torch.sigmoid(x @ m.in_proj_b.weight.T)
    g = -m.A_log.exp() * F.softplus(x @ m.in_proj_a.weight.T + m.dt_bias)
    H, Dk, Dv = v.shape[2], k.shape[-1], v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv)
    o = torch.zeros(B, T, H, Dv)
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - pred) * beta[:, t, :, None])
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    z = (x @ m.in_proj_z.weight.T).reshape(B, T, H, Dv)
    o = g_rms(o, m.norm.weight, cfg.rms_norm_eps, unit_offset=False) * F.silu(z)
    return o.reshape(B, T, -1) @ m.out_proj.weight.T, S, qkv[:, -(K - 1) :, :].transpose(1, 2)


def g_attn(m, x, cfg, pos):
    B, T, _ = x.shape
    hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    qg = (x @ m.q_proj.weight.T).reshape(B, T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(B, T, -1)
    k = (x @ m.k_proj.weight.T).reshape(B, T, nkv, hd)
    v = (x @ m.v_proj.weight.T).reshape(B, T, nkv, hd)
    q, k = g_rms(q, m.q_norm.weight, cfg.rms_norm_eps), g_rms(k, m.k_norm.weight, cfg.rms_norm_eps)
    rd = cfg.rotary_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
    ang = pos.float()[:, None] * inv[None]
    c, s = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]

    def rot(t):
        a, b = t[..., : rd // 2], t[..., rd // 2 : rd]
        return torch.cat([a * c - b * s, b * c + a * s, t[..., rd:]], -1)

    q, k = rot(q), rot(k)
    out = torch.zeros(B, T, nh, hd)
    for h in range(nh):
        kv = h // (nh // nkv)
        sc = torch.einsum("btd,bsd->bts", q[:, :, h], k[:, :, kv]) / hd**0.5
        sc = sc.masked_fill(torch.ones(T, T).triu(1).bool(), float("-inf"))
        out[:, :, h] = torch.softmax(sc, -1) @ v[:, :, kv]
    out = out.reshape(B, T, -1) * torch.sigmoid(gate)
    return out @ m.o_proj.weight.T, k.transpose(1, 2), v.transpose(1, 2)


def g_model(model, ids, cfg):
    x = model.embed_tokens.weight[ids].float()
    T = ids.shape[1]
    states = []
    for L in model.layers:
        h = g_rms(x, L.input_layernorm.weight, cfg.rms_norm_eps)
        if L.is_full:
            h, k, v = g_attn(L.self_attn, h, cfg, torch.arange(T))
            states.append({"k": k, "v": v})
        else:
            h, S, cs = g_gdn(L.linear_attn, h, cfg)
            states.append({"recurrent_state": S, "conv_state": cs})
        x = x + h
        m = L.mlp
        hh = g_rms(x, L.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x = x + (F.silu(hh @ m.gate_proj.weight.T) * (hh @ m.up_proj.weight.T)) @ m.down_proj.weight.T
    return g_rms(x, model.norm.weight, cfg.rms_norm_eps), states


@pytest.fixture(scope="module")
def model():
    return ref.init_random_(ref.TextModel(SMALL), seed=3).float().eval()


def test_reference_vs_inline_golden(model):
    torch.manual_seed(0)
    ids = torch.randint(0, SMALL.vocab_size, (1, 160))
    with torch.no_grad():
        h, st = model(ids)
        gh, gst = g_model(model, ids, SMALL)
    assert pcc(h, gh) > 0.9999
    for i, (a, b) in enumerate(zip(st, gst)):
        for key in a:
            assert pcc(a[key], b[key]) > 0.9999, (i, key)


def test_reference_chunked_equals_one_shot(model):
    """Two chunks with carried state == one shot (what P2 relies on)."""
    torch.manual_seed(1)
    ids = torch.randint(0, SMALL.vocab_size, (1, 256))
    with torch.no_grad():
        h1, st1 = model(ids)
        ha, sta = model(ids[:, :128])
        hb, stb = model(ids[:, 128:], states=sta, start_pos=128)
    assert pcc(torch.cat([ha, hb], 1), h1) > 0.99999
    for a, b in zip(stb, st1):
        for key in a:
            assert pcc(a[key], b[key]) > 0.99999, key


def test_reference_bf16_close_to_fp32(model):
    """The bf16 reference (the recipe convention) stays close to fp32 on the reduced model."""
    torch.manual_seed(2)
    ids = torch.randint(0, SMALL.vocab_size, (1, 128))
    with torch.no_grad():
        h32, _ = model(ids)
        h16, _ = model.to(torch.bfloat16)(ids)
    model.float()
    assert pcc(h16.float(), h32) > 0.99
