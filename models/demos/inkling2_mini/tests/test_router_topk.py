# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused decode router top-k (tt/fused/router_topk.py) vs ttnn moe_grouped_topk as tt/moe.py calls it:
ids (with order) and bf16 weights bit-equal over random logits, exact ties and near-ties (keys equal at 17
mantissa bits), plus traced replay time of 48 calls each.
"""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.router_topk import FusedRouterTopk
from models.demos.inkling2_mini.tt.moe import NUM_EXPERTS, ROUTE_SCALE, TOP_K

TRACE_PARAMS = [{"trace_region_size": 8 * 1024 * 1024}]


def op(logits, bias):
    return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        logits, bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=TOP_K,
        route_scale=ROUTE_SCALE, epsilon=1e-20, stable_sort=True, memory_config=ttnn.L1_MEMORY_CONFIG,
    )  # fmt: skip


def host(w, i):
    return ttnn.to_torch(w).reshape(-1)[:TOP_K].view(torch.int16), ttnn.to_torch(i).reshape(-1)[:TOP_K].int()


def replay_us(dev, tid, reps=50):
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e6


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_router_topk(device):
    dev = device
    g = torch.Generator().manual_seed(0)
    bias = torch.randn(NUM_EXPERTS, generator=g) * 0.05
    bias = (bias - bias.mean()).reshape(1, 1, 1, NUM_EXPERTS)
    put = lambda t: ttnn.from_torch(  # noqa: E731
        t.reshape(1, 1, 1, NUM_EXPERTS), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )  # fmt: skip
    bias_t = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    fused = FusedRouterTopk(dev, bias_t, TOP_K, ROUTE_SCALE, 1e-20)
    zero_bias = ttnn.from_torch(torch.zeros_like(bias), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    fused0 = FusedRouterTopk(dev, zero_bias, TOP_K, ROUTE_SCALE, 1e-20)

    cases = []
    for _ in range(60):
        cases.append(("random", torch.randn(NUM_EXPERTS, generator=g) * 2.0, bias_t, fused))
    for _ in range(20):  # exact ties among the top keys (zero bias: equal logits = equal keys)
        v = torch.randn(NUM_EXPERTS, generator=g)
        top = torch.randperm(NUM_EXPERTS, generator=g)[:12]
        v[top] = 3.0
        cases.append(("exact_tie", v, zero_bias, fused0))
    for _ in range(20):  # near ties: sigmoid values differ only in the low mantissa bits
        v = torch.randn(NUM_EXPERTS, generator=g)
        top = torch.randperm(NUM_EXPERTS, generator=g)[:12]
        v[top] = 2.0 + torch.arange(12) * 1e-6
        cases.append(("near_tie", v, zero_bias, fused0))
    bad = 0
    for name, v, b, f in cases:
        lt = put(v)
        rw, ri = host(*op(lt, b))
        fw, fi = host(*f(lt))
        if not (torch.equal(ri, fi) and torch.equal(rw, fw)):
            bad += 1
            print(f"MISMATCH {name}: ids op {ri.tolist()} fused {fi.tolist()} w op {rw.tolist()} fused {fw.tolist()}")
    print(f"ROUTER_CASES {len(cases)} mismatches {bad}")
    assert bad == 0

    lt = put(cases[0][1])
    times = {}
    for label, fn in (("op", lambda: op(lt, bias_t)), ("fused", lambda: fused(lt))):
        fn()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(48):
            fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        times[label] = replay_us(dev, tid) / 48
        ttnn.release_trace(dev, tid)
    print(f"ROUTER_US per call op {times['op']:.1f} fused {times['fused']:.1f}")
