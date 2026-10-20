# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops.kda_decode`` vs its torch golden: the model's shapes (K = V = 128, 64 heads on one chip and the TP4
shard's 16 heads), B = 1 / 4 / 32 (and a two-tile-row case), multi-step evolution of the in-place state and conv history,
and a captured-trace replay on persistent io tensors.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_kda_decode.py -s
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.glm53_flash.custom_ops.kda_decode.op import KdaDecode, selector_tiles
from models.demos.glm53_flash.tests.utils import pcc

K = V = 128
LB, L2_EPS, NORM_EPS = -5.0, 1e-6, 1e-5
PCC = 0.9999


def _dev(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_inputs(B, H, seed):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    W = 3 * H * K
    Hb = -(-H // 32) * 32
    qkv = r(1, 1, B, W).to(torch.bfloat16).float()
    gates = torch.zeros(1, 1, B, H * K + H * V + Hb)
    gates[..., : H * K] = r(B, H * K) * 2.0
    gates[..., H * K : H * K + H * V] = r(B, H * V) * 2.0
    gates[..., H * K + H * V : H * K + H * V + H] = r(B, H) * 2.0
    conv = (r(B, 3, W) * 0.8).to(torch.bfloat16).float()
    S = 0.05 * r(B, H, K, V)
    taps = r(4, W) * 0.5
    dt_bias = r(H * K) * 0.5
    a_exp = torch.rand(H, generator=gen).repeat_interleave(K) * 4.0 + 0.5
    onw = torch.rand(V, generator=gen) * 0.5 + 0.75
    return dict(qkv=qkv, gates=gates, conv=conv, S=S, taps=taps, dt_bias=dt_bias, a_exp=a_exp, onw=onw)


def to_device(device, x, B, H):
    return dict(
        qkv=_dev(device, x["qkv"].to(torch.bfloat16), ttnn.bfloat16),
        gates=_dev(device, x["gates"]),
        conv=_dev(device, x["conv"].to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        S=_dev(device, x["S"]),
        taps=_dev(device, x["taps"].reshape(1, 1, 4, -1)),
        cc=_dev(device, torch.stack([x["dt_bias"], x["a_exp"]]).reshape(1, 1, 2, -1)),
        onw=_dev(device, x["onw"].reshape(1, 1, 1, -1)),
        sel=_dev(device, selector_tiles().to(torch.bfloat16), ttnn.bfloat16),
        out=_dev(device, torch.zeros(1, 1, B, H * V).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    )


def run(op, d):
    return op(d["qkv"], d["gates"], d["conv"], d["S"], d["taps"], d["cc"], d["onw"], d["sel"], d["out"])


def golden(x, B, H):
    return KdaDecode.golden(
        x["qkv"][0, 0, :B],
        x["gates"][0, 0, :B],
        x["conv"],
        x["S"],
        x["taps"],
        x["dt_bias"],
        x["a_exp"],
        x["onw"],
        num_heads=H,
        gate_lower_bound=LB,
        l2_eps=L2_EPS,
        norm_eps=NORM_EPS,
    )


def check(d, x, B, H, o_ref, S_ref, conv_ref):
    o = ttnn.to_torch(d["out"]).float()[0, 0]
    S = ttnn.to_torch(d["S"])
    conv = ttnn.to_torch(d["conv"]).float()
    assert o.shape == (B, H * V)
    p_o, p_s = pcc(o_ref, o), pcc(S_ref, S)
    assert p_o > PCC and p_s > PCC, (p_o, p_s)
    torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)  # bf16 rows moved, not recomputed
    assert (o - o_ref).abs().max() < 0.05 and (S - S_ref).abs().max() < 0.05  # FPU precision, O(1) values
    return p_o, p_s


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B, H", [(1, 64), (4, 64), (32, 64), (4, 16), (32, 16), (40, 16)])
def test_kda_decode_vs_golden(mesh_device, B, H):
    op = KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS)
    x = make_inputs(B, H, seed=B * 3 + H)
    refs = golden(x, B, H)
    d = to_device(mesh_device, x, B, H)
    run(op, d)
    p_o, p_s = check(d, x, B, H, *refs)
    print(f"B={B} H={H}: o pcc {p_o:.6f} state pcc {p_s:.6f}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_decode_multi_step(mesh_device):
    B, H = 4, 64
    op = KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS)
    x = make_inputs(B, H, seed=7)
    d = to_device(mesh_device, x, B, H)
    ref = dict(x)
    for step in range(4):
        gen = torch.Generator().manual_seed(100 + step)
        ref["qkv"] = torch.randn(1, 1, B, 3 * H * K, generator=gen).to(torch.bfloat16).float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(ref["qkv"].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), d["qkv"]
        )
        o_ref, S_ref, conv_ref = golden(ref, B, H)
        ref["S"], ref["conv"] = S_ref, conv_ref
        run(op, d)
        check(d, ref, B, H, o_ref, S_ref, conv_ref)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_decode_trace_replay(mesh_device):
    """The program inside a captured trace on persistent io tensors: the capture run and three replays advance
    the state like four eager calls (addresses are baked into the runtime args)."""
    B, H = 4, 16
    op = KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS)
    x = make_inputs(B, H, seed=5)
    d = to_device(mesh_device, x, B, H)
    ref = dict(x)
    run(op, d)  # compile (advances the state once)
    o_ref, S_ref, conv_ref = golden(ref, B, H)
    ref["S"], ref["conv"] = S_ref, conv_ref
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run(op, d)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    # the capture run executes as well (the state advances), then 3 replays; 4 golden steps on the same qkv
    for _ in range(4):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        o_ref, S_ref, conv_ref = golden(ref, B, H)
        ref["S"], ref["conv"] = S_ref, conv_ref
    check(d, ref, B, H, o_ref, S_ref, conv_ref)
    ttnn.release_trace(mesh_device, tid)
    print("trace replay: ok")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B, H", [(4, 64), (32, 64), (4, 16), (32, 16)])
def test_kda_decode_perf(mesh_device, B, H):
    import time

    op = KdaDecode(num_heads=H, gate_lower_bound=LB, l2_eps=L2_EPS, norm_eps=NORM_EPS)
    d = to_device(mesh_device, make_inputs(B, H, seed=1), B, H)
    for _ in range(3):
        run(op, d)
    ttnn.synchronize_device(mesh_device)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        run(op, d)
    ttnn.synchronize_device(mesh_device)
    print(f"kda_decode B={B} H={H}: {(time.perf_counter() - t0) / n * 1e6:.0f} us/call (eager)")
