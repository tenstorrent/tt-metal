# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused decode router (tt/fused/router.py) vs the op chain it replaces in tt/moe.py route() (reshape, K-split
matmul, ttnn.sum, moe_grouped_topk): ids (with order) and bf16 weights bit-equal over random inputs, exact ties
(duplicate weight columns) and near ties, plus traced replay time of 48 calls each.
"""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.router import FusedRouter
from models.demos.inkling2_mini.tt.moe import HIDDEN, NUM_EXPERTS, ROUTE_SCALE, ROUTER_SPLIT, TOP_K

TRACE_PARAMS = [{"trace_region_size": 8 * 1024 * 1024}]


def replay_us(dev, tid, reps=50):
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e6


class Chain:
    """tt/moe.py route() before fusion."""

    def __init__(self, dev, w, bias):
        self.w = ttnn.from_torch(
            w.reshape(1, ROUTER_SPLIT, HIDDEN // ROUTER_SPLIT, NUM_EXPERTS), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=dev,
        )  # fmt: skip
        self.bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, NUM_EXPERTS), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev
        )
        self.cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )  # fmt: skip

    def __call__(self, x):
        xk = ttnn.reshape(x, (1, ROUTER_SPLIT, 1, HIDDEN // ROUTER_SPLIT))
        p = ttnn.matmul(
            xk, self.w, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=self.cc
        )
        logits = ttnn.sum(p, dim=1, keepdim=True)
        return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits, self.bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=TOP_K,
            route_scale=ROUTE_SCALE, epsilon=1e-20, stable_sort=True, memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # fmt: skip


def host(w, i):
    return ttnn.to_torch(w).reshape(-1)[:TOP_K].view(torch.int16), ttnn.to_torch(i).reshape(-1)[:TOP_K].int()


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_router(device):
    dev = device
    g = torch.Generator().manual_seed(0)
    w = torch.randn(HIDDEN, NUM_EXPERTS, generator=g) * 0.02
    dup = torch.randperm(NUM_EXPERTS, generator=g)[:16]  # exact ties: 8 pairs of identical columns
    w[:, dup[8:]] = w[:, dup[:8]]
    w[:, dup[0]] += 1e-7  # near tie
    bias = torch.randn(NUM_EXPERTS, generator=g) * 0.05
    bias = bias - bias.mean()
    bias[dup[8:]] = bias[dup[:8]]
    chain = Chain(dev, w, bias)
    fused = FusedRouter(dev, chain.w, bias, TOP_K, ROUTE_SCALE, 1e-20)

    def put(v):
        return ttnn.from_torch(
            v.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # fmt: skip

    bad, n = 0, 0
    for t in range(80):
        x = torch.randn(HIDDEN, generator=g) * (1.0 if t < 40 else 0.1)
        if t >= 60:  # steer the duplicate columns into the top-k
            d = w[:, dup[:8]].sum(1)
            x = x + 3.0 * d / d.norm()
        xt = put(x)
        rw, ri = host(*chain(xt))
        fw, fi = host(*fused(xt))
        n += 1
        if not (torch.equal(ri, fi) and torch.equal(rw, fw)):
            bad += 1
            print(f"MISMATCH {t}: ids chain {ri.tolist()} fused {fi.tolist()} w {rw.tolist()} vs {fw.tolist()}")
    print(f"ROUTER_CASES {n} mismatches {bad}")
    assert bad == 0

    xt = put(torch.randn(HIDDEN, generator=g))
    times = {}
    for label, fn in (("chain", chain), ("fused", fused)):
        fn(xt)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(48):
            fn(xt)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        times[label] = replay_us(dev, tid) / 48
        ttnn.release_trace(dev, tid)
    print(f"ROUTER_US per call chain {times['chain']:.1f} fused {times['fused']:.1f}")
