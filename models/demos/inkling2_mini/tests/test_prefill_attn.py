# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention core, block test (seconds, no model load), random inputs, local and global layers, first chunk
and later chunks (ring rotation, s < 1024, far keys): the R3 path (one SDPA with in-kernel banded rel bias and
per-query sink) and the blocked path (bias-matmul mask, near/far split; builds without R3) against a torch reference.

    pytest -s models/demos/inkling2_mini/tests/test_prefill_attn.py
"""

import types

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.attention_fast import KV_WIDTH, V_WIDTH
from models.demos.inkling2_mini.tt.prefill_model import (
    D_REL,
    HEAD_DIM,
    N_HEADS,
    N_KV,
    WINDOW,
    Inkling2Prefill,
    _band_matrix,
)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def _pf(dev, max_ctx, st, r3):
    pf = object.__new__(Inkling2Prefill)
    pf.device = dev
    pf.m = types.SimpleNamespace(max_ctx=max_ctx, states=[(st,)])
    pf.layers, pf.chunk, pf.r3 = [], 2048, r3
    pf.hifi4 = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    pf.sdpa_ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    pf._sdpa_pc = {}
    pf.r3_chunks = (128, 128)
    pf.profile = False
    # the fused programs __init__ sets up stay off here: plain ttnn ops for the rel bias, ring write and heads
    pf._pfrel = pf._pfring = pf._pfsink = pf._pfgate = pf._pfheads = None
    pf.fused_rel = False
    pf._cap = pf._cur = pf._mem = None
    pf._consts()
    return pf


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def _reference(qs, k_pos, v_pos, rel, sink, p, s, is_global):
    """Queries at positions s.., keys at 0..s+t: logit q.k + rel . p[:, d] (d = q_pos - k_pos < 1024); d < 0 masked;
    local: d >= 1024 masked; sink: extra logit with zero value. GQA: head h reads kv head h // 3."""
    t, n = qs.shape[2], k_pos.shape[2]
    k = k_pos.repeat_interleave(N_HEADS // N_KV, dim=1)
    v = v_pos.repeat_interleave(N_HEADS // N_KV, dim=1)
    lg = qs @ k.transpose(-1, -2)  # [1, 24, t, n]
    d = (s + torch.arange(t)).view(t, 1) - torch.arange(n).view(1, n)
    bias = rel @ p  # [1, 24, t, 1024]
    lg = lg + torch.where((d >= 0) & (d < WINDOW), torch.gather(bias, -1, d.clamp(0, WINDOW - 1).expand_as(lg)), 0.0)
    vis = d >= 0
    if not is_global:
        vis &= d < WINDOW
    lg = lg.masked_fill(~vis, float("-inf"))
    lg = torch.cat([lg, sink], dim=-1)
    w = torch.softmax(lg, dim=-1)[..., :n]
    return w @ v


@pytest.mark.parametrize("path", ["r3", "blocked"])
@pytest.mark.parametrize("is_global", [False, True], ids=["local", "global"])
@pytest.mark.parametrize(
    "s, t", [(0, 256), (768, 256), (1280, 256), (2048, 2048)], ids=["first", "s768", "s1280", "later"]
)
def test_attention_core(device, path, is_global, s, t):
    if path == "r3" and "rel_logits" not in (ttnn.transformer.scaled_dot_product_attention.__doc__ or ""):
        pytest.skip("this build's SDPA has no rel_logits (R3)")
    torch.manual_seed(0)
    max_ctx = 4096 + 128
    put = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)  # noqa
    bf = lambda x: x.bfloat16().float()  # noqa: E731
    n = s + t
    k_pos = bf(torch.nn.functional.normalize(torch.randn(1, N_KV, n, HEAD_DIM), dim=-1) * 4)
    v_pos = bf(torch.randn(1, N_KV, n, HEAD_DIM))
    qs = bf(torch.nn.functional.normalize(torch.randn(1, N_HEADS, t, HEAD_DIM), dim=-1) * 2)
    rel = bf(torch.randn(1, N_HEADS, t, D_REL))
    sink = bf(torch.randn(1, N_HEADS, t, 1) * 2)
    p = bf(torch.randn(D_REL, WINDOW) * 0.3)
    lp = types.SimpleNamespace(
        rel_proj=put(p.view(1, 1, D_REL, WINDOW)),
        bu=put(_band_matrix(p, is_global)) if path == "blocked" else None,
        lay=types.SimpleNamespace(is_global=is_global),
    )
    ring_k = torch.zeros(1, N_KV, WINDOW, HEAD_DIM)
    ring_v = torch.zeros(1, N_KV, WINDOW, HEAD_DIM)
    for pos in range(max(0, s - WINDOW), s):  # slot = pos % 1024 holds the last 1024 positions before the chunk
        ring_k[:, :, pos % WINDOW] = k_pos[:, :, pos]
        ring_v[:, :, pos % WINDOW] = v_pos[:, :, pos]
    kv = torch.zeros(1, N_KV, max_ctx, KV_WIDTH)
    kv[:, :, :n, :HEAD_DIM] = v_pos
    kv[:, :, :n, HEAD_DIM] = 1
    kv[:, :, :n, V_WIDTH:] = k_pos
    st = types.SimpleNamespace(k_ring=put(ring_k), v_ring=put(ring_v), kv_full=put(kv))
    pf = _pf(device, max_ctx, st, path == "r3")
    args = (lp, put(qs), put(rel), put(sink), st, put(k_pos[:, :, s:]), put(v_pos[:, :, s:]), s, t)
    if path == "r3":
        out = pf._attention_r3(*args)
    else:
        out = pf._global_blocked(*args) if is_global else pf._local_blocked(*args)
    out = ttnn.to_torch(out).float()
    ref = _reference(qs, k_pos, v_pos, rel, sink, p, s, is_global)
    pcc, err = _pcc(out, ref), float((out - ref).abs().max())
    print(f"{path} {'global' if is_global else 'local'} s={s} t={t}: pcc {pcc:.6f} max abs diff {err:.4f}", flush=True)
    assert pcc >= 0.9999
