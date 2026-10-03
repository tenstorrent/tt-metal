# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSV41_ROUTER=fused (_forward_fused, JIT router_select) vs _forward_exact_fast2 / exact_select: ids identical, weights <= 1 bf16 ulp,
on random rows (T=4/8/16/32), real layer inputs, forced ties; and trace timing per call."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_router_v2 import (
    CHAIN,
    PARAMS,
    Acc,
    chain_ms,
    d0,
    make_gate,
    up,
)
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_fused_exact(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 32)
    torch.manual_seed(1)
    tot = Acc()
    for T, iters in ((32, 150), (16, 100), (8, 100), (4, 100)):
        acc = Acc()
        for i in range(iters):
            x = torch.randn(T, 5120) * (0.25, 0.5, 1.0, 2.0, 4.0)[i % 5]
            if i % 7 == 0:
                x = x * (torch.rand(1, 5120) < 0.05) * 8 + x * 0.3
            if i % 11 == 0 and T > 1:
                x[T // 2] = x[0]  # identical rows -> identical ranks, different rows exercised
            h = up(md, x)
            wa, ia = gate._forward_exact_fast2(h, exact_select=True)
            wb, ib = gate._forward_fused(h)
            acc.add(d0(ia, T).long(), d0(wa, T), d0(ib, T).long(), d0(wb, T))
            ttnn.deallocate(h)
        print(f"FUSED random T={T}: {acc}", flush=True)
        tot.rows += acc.rows
        tot.id_eq += acc.id_eq
        tot.set_eq += acc.set_eq
        tot.wmax = max(tot.wmax, acc.wmax)
        tot.ulpmax = max(tot.ulpmax, acc.ulpmax)
        tot.nulp += acc.nulp
    print(f"FUSED random TOTAL: {tot}", flush=True)
    assert tot.id_eq == tot.rows
    assert tot.ulpmax <= 1.01
    for L in (1, 2, 8, 14, 20):
        p = os.path.join(CHAIN, f"ffn_inputs_{L}.pt")
        if not os.path.exists(p):
            continue
        d = torch.load(p)
        x = d["x"].reshape(16, 5120)
        wl = load_moe_layer(L, experts=[0])
        W, b = wl["gate_weight"].float(), wl["gate_bias"].float()
        rk = torch.nn.functional.softplus(x.float() @ (W.T if W.shape[0] == 384 else W)).sqrt() + b
        for shift in (0.0, float(rk.topk(7, -1).values[:, 5].mean())):
            g, _ = make_gate(md, L, 16, shift)
            h = up(md, x)
            wa, ia = g._forward_exact_fast2(h, exact_select=True)
            wb, ib = g._forward_fused(h)
            acc = Acc()
            acc.add(d0(ia, 16).long(), d0(wa, 16), d0(ib, 16).long(), d0(wb, 16))
            print(f"FUSED real layer {L} shift {shift:.3f}: {acc}", flush=True)
            assert acc.id_eq == 16 and acc.ulpmax <= 1.01


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_fused_timing(mesh_device):
    md = mesh_device
    for T in (4, 8, 16, 32):
        gate, _ = make_gate(md, 2, T)
        h = up(md, torch.randn(T, 5120))
        for n, fn in (
            ("exact_fast2 (default)", lambda: gate._forward_exact_fast2(h)),
            ("fused", lambda: gate._forward_fused(h)),
        ):
            print(f"FUSEDT T={T:2d} {n:24s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
