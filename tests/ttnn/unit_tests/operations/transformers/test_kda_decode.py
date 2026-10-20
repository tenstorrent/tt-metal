# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of ttnn.transformer.kda_decode (the fused KDA single-token decode block) against a torch reference of
the block: conv (kernel 4) + SiLU over the history, per-head L2 norms, per-key-channel decay / beta / output gates, the
gated delta-rule step on the state and the gated RMSNorm; the state and the conv history are updated in place.

    pytest tests/ttnn/unit_tests/operations/transformers/test_kda_decode.py -s
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

OP = ttnn.transformer.kda_decode
K = V = 128
LB, L2_EPS, NORM_EPS = -5.0, 1e-6, 1e-5
PCC = 0.9999


def _dev(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def selectors():
    sel = torch.zeros(33, 32, 32)
    sel[torch.arange(32), 0, torch.arange(32)] = 1.0
    sel[32] = 1.0
    return sel.reshape(33 * 32, 32)


def make_inputs(B, H, seed):
    """One decode step's inputs in the block's contract, realistic scales (bf16-rounded where the contract is bf16)."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    W = 3 * H * K
    Hb = -(-H // 32) * 32
    qkv = r(1, 1, B, W).to(torch.bfloat16).float()
    gates = torch.zeros(1, 1, B, H * K + H * V + Hb)
    gates[..., : H * K] = r(B, H * K) * 2.0  # forget-gate pre-activation
    gates[..., H * K : H * K + H * V] = r(B, H * V) * 2.0  # output gate
    gates[..., H * K + H * V : H * K + H * V + H] = r(B, H) * 2.0  # beta logits
    conv = (r(B, 3, W) * 0.8).to(torch.bfloat16).float()
    S = 0.05 * r(B, H, K, V)
    taps = r(4, W) * 0.5
    dt_bias = r(H * K) * 0.5
    a_exp = torch.rand(H, generator=gen).repeat_interleave(K) * 4.0 + 0.5  # exp(A_log) per channel
    onw = torch.rand(V, generator=gen) * 0.5 + 0.75
    return dict(qkv=qkv, gates=gates, conv=conv, S=S, taps=taps, dt_bias=dt_bias, a_exp=a_exp, onw=onw)


def kda_block_ref(x, B, H):
    """Torch reference of one step; returns (o [B, H*V], new state, new conv history)."""
    qkv, gates, conv, S = x["qkv"][0, 0, :B], x["gates"][0, 0, :B], x["conv"], x["S"]
    taps = x["taps"]
    rows = torch.cat([conv, qkv[:, None, :]], dim=1)  # [B, 4, W], oldest first
    cv = torch.nn.functional.silu((rows * taps[None]).sum(1))  # [B, W]
    q, k, v = (t.reshape(B, H, K) for t in cv.split(H * K, dim=-1))
    l2 = lambda t: t / torch.sqrt((t * t).sum(-1, keepdim=True) + L2_EPS)
    q = l2(q) * K**-0.5
    k = l2(k)
    f = gates[:, : H * K]
    og = gates[:, H * K : H * K + H * V].reshape(B, H, V)
    b = gates[:, H * K + H * V : H * K + H * V + H]
    decay = torch.exp(LB * torch.sigmoid(x["a_exp"] * (f + x["dt_bias"]))).reshape(B, H, K)
    beta = torch.sigmoid(b)
    gate = torch.sigmoid(og)
    Sd = S * decay[..., None]
    vread = (Sd * k[..., None]).sum(-2)
    u = beta[..., None] * (v - vread)
    Sn = Sd + k[..., None] * u[:, :, None, :]
    o = (Sn * q[..., None]).sum(-2)  # [B, H, V]
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + NORM_EPS) * x["onw"] * gate
    new_conv = torch.cat([conv[:, 1:], qkv[:, None, :]], dim=1)
    return o.reshape(B, H * V), Sn, new_conv


def _to_device(device, x, B):
    return dict(
        qkv=_dev(device, x["qkv"].to(torch.bfloat16), ttnn.bfloat16),
        gates=_dev(device, x["gates"]),
        conv=_dev(device, x["conv"].to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        S=_dev(device, x["S"]),
        taps=_dev(device, x["taps"].reshape(1, 1, 4, -1)),
        cc=_dev(device, torch.stack([x["dt_bias"], x["a_exp"]]).reshape(1, 1, 2, -1)),
        onw=_dev(device, x["onw"].reshape(1, 1, 1, -1)),
        sel=_dev(device, selectors().to(torch.bfloat16), ttnn.bfloat16),
    )


def _run(d, H):
    return OP(
        d["qkv"],
        d["gates"],
        d["conv"],
        d["S"],
        d["taps"],
        d["cc"],
        d["onw"],
        d["sel"],
        H,
        gate_lower_bound=LB,
        l2_eps=L2_EPS,
        norm_eps=NORM_EPS,
    )


@pytest.mark.parametrize(
    "B, H",
    [
        (4, 64),  # GLM-5.3 single chip, 4 users
        (32, 64),  # 32 users (full tile row), 64 work items
        (40, 16),  # two tile rows, the second partial (TP4 head count)
        (1, 16),
    ],
)
def test_kda_decode_vs_reference(device, B, H):
    x = make_inputs(B, H, seed=B * 3 + H)
    o_ref, S_ref, conv_ref = kda_block_ref(x, B, H)
    d = _to_device(device, x, B)
    o = _run(d, H)
    o_t = ttnn.to_torch(o).float()[0, 0]
    S_t = ttnn.to_torch(d["S"])
    conv_t = ttnn.to_torch(d["conv"]).float()
    assert o_t.shape == (B, H * V)
    assert_with_pcc(o_ref, o_t, PCC)
    assert_with_pcc(S_ref, S_t, PCC)
    torch.testing.assert_close(conv_t, conv_ref, rtol=0, atol=0)  # exact: bf16 rows moved, not recomputed
    err_o = (o_t - o_ref).abs().max().item()
    err_s = (S_t - S_ref).abs().max().item()
    print(f"B={B} H={H}: max|o-ref| {err_o:.3e} max|S-ref| {err_s:.3e}")
    assert err_o < 0.05 and err_s < 0.02


def test_kda_decode_multi_step(device):
    """Four consecutive steps on the in-place state and history match four reference steps."""
    B, H = 4, 64
    x = make_inputs(B, H, seed=7)
    d = _to_device(device, x, B)
    ref = dict(x)
    for step in range(4):
        gen = torch.Generator().manual_seed(100 + step)
        qkv = torch.randn(1, 1, B, 3 * H * K, generator=gen).to(torch.bfloat16).float()
        ref["qkv"] = qkv
        d["qkv"] = _dev(device, qkv.to(torch.bfloat16), ttnn.bfloat16)
        o_ref, S_ref, conv_ref = kda_block_ref(ref, B, H)
        ref["S"], ref["conv"] = S_ref, conv_ref
        o = _run(d, H)
        o_t = ttnn.to_torch(o).float()[0, 0]
        assert_with_pcc(o_ref, o_t, PCC)
        assert_with_pcc(S_ref, ttnn.to_torch(d["S"]), PCC)
        torch.testing.assert_close(ttnn.to_torch(d["conv"]).float(), conv_ref, rtol=0, atol=0)
        print(f"step {step}: ok")


@pytest.mark.parametrize("B, H", [(4, 64), (32, 64), (4, 16), (32, 16)])
def test_kda_decode_perf(device, B, H):
    """Op-level latency (eager, 20 iterations after warm-up): B users x H local heads (H=16 is the TP4 shard)."""
    import time

    x = make_inputs(B, H, seed=1)
    d = _to_device(device, x, B)
    for _ in range(3):
        _run(d, H)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        _run(d, H)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / n
    print(f"kda_decode B={B} H={H}: {dt*1e6:.0f} us/call (eager)")
