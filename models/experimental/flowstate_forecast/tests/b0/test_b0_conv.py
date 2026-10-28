# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""B0 accuracy and timing of the convolution routes on chip (CONTRACT v1.1 §6 B0, D0 items 3, 4, 6, 7).

Accuracy: every B0 case (L = 682, 4096, 16384 x2) and captured layer (0, 3, 4, 5). Timing: one layer at paper
(64 x 4096, s = 1) and long_context (16 x 16384, s = 0.25) on random fp32 Bu. Results go to B0_OUT as json.
"""

import os

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.tests.b0 import route_f
from models.experimental.flowstate_forecast.tests.b0.b0_common import B0_CASES, case_inputs, dump, median_ms, metrics

OUT = os.environ.get("B0_OUT", "/tmp/flowstate_b0")
LAYERS = (0, 3, 4, 5)


def to_dev(device, t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def run_f(device, Bu, K, N=None):
    L = Bu.shape[0]
    N = N or route_f.pow2_n(L)
    kh = route_f.spectrum(device, K, N)
    xr, xi = route_f.conv(to_dev(device, Bu.real), to_dev(device, Bu.imag), kh, N)
    return torch.complex(ttnn.to_torch(xr).float(), ttnn.to_torch(xi).float())


@pytest.mark.parametrize("case", B0_CASES)
def test_route_f_accuracy(device, case):
    res = {}
    for layer in LAYERS:
        c = case_inputs(case, layer)
        res[layer] = metrics(run_f(device, c.Bu, c.K), c.ref, c.truth)
        print("F", case, layer, res[layer], flush=True)
    dump(f"{OUT}/F_acc_{case}.json", res)


def test_route_f_control(device):
    """Positive control (D0 item 4): a circular convolution at N = L must fail the bound."""
    c = case_inputs("b0_s1_L4096", 0)
    m = metrics(run_f(device, c.Bu, c.K, N=c.L), c.ref, c.truth)
    print("F control circular N=L", m)
    dump(f"{OUT}/F_control.json", m)
    assert m["rel_l2"] > 1e-3


@pytest.mark.parametrize("shape", [(64, 4096), (16, 16384)], ids=["paper", "long_context"])
def test_route_f_time(device, shape):
    B, L = shape
    N = route_f.pow2_n(L)
    g = torch.Generator().manual_seed(0)
    bu_re = to_dev(device, torch.randn(L, B, 512, generator=g))
    bu_im = to_dev(device, torch.randn(L, B, 512, generator=g))
    K = torch.complex(torch.randn(L, 512), torch.randn(L, 512)) * 0.01
    kh = route_f.spectrum(device, K, N)

    def fn():
        out = route_f.conv(bu_re, bu_im, kh, N)
        for t in out:
            ttnn.deallocate(t)

    ms = median_ms(fn, lambda: ttnn.synchronize_device(device))
    print("F time", shape, ms)
    dump(f"{OUT}/F_time_B{B}_L{L}.json", {"B": B, "L": L, "N": N, "ms": ms, "timing": "synchronized wall, untraced"})


# ---------------------------------------------------------------- route S

from models.experimental.flowstate_forecast.tests.b0 import route_s  # noqa: E402


def run_s(device, Bu, lbar, c):
    L = Bu.shape[0]
    Lp = -(-L // c) * c
    Bu = torch.cat((Bu, torch.zeros(Lp - L, *Bu.shape[1:], dtype=Bu.dtype)))
    tb = route_s.tables(device, lbar, c, Lp // c, Bu.abs().max().item())
    xr, xi = route_s.conv(to_dev(device, Bu.real), to_dev(device, Bu.imag), tb, c)
    return torch.complex(ttnn.to_torch(xr).float(), ttnn.to_torch(xi).float())[:L], int(tb["bounded"].sum())


@pytest.mark.parametrize("c", [32, 64])
@pytest.mark.parametrize("case", B0_CASES)
def test_route_s_accuracy(device, case, c):
    res = {}
    for layer in LAYERS:
        cs = case_inputs(case, layer)
        x, nb = run_s(device, cs.Bu, cs.lbar, c)
        res[layer] = {**metrics(x, cs.ref, cs.truth), "bounded_modes": nb}
        print("S", c, case, layer, res[layer], flush=True)
    dump(f"{OUT}/S{c}_acc_{case}.json", res)


def test_route_s_control(device):
    """Positive control (D0 item 4): a bf16 state must fail the bound."""
    cs = case_inputs("b0_s1_L4096", 4)
    x, _ = run_s(device, cs.Bu, cs.lbar, 32)
    x = torch.complex(x.real.bfloat16().float(), x.imag.bfloat16().float())
    m = metrics(x, cs.ref, cs.truth)
    print("S control bf16 state", m)
    dump(f"{OUT}/S_control.json", m)
    assert m["rel_l2"] > 1e-3


@pytest.mark.parametrize("c", [32, 64])
@pytest.mark.parametrize("shape", [(64, 4096), (16, 16384)], ids=["paper", "long_context"])
def test_route_s_time(device, shape, c):
    B, L = shape
    g = torch.Generator().manual_seed(0)
    bu_re = to_dev(device, torch.randn(L, B, 512, generator=g))
    bu_im = to_dev(device, torch.randn(L, B, 512, generator=g))
    lbar = torch.exp(torch.complex(-torch.rand(512) * 0.1, torch.rand(512)))
    tb = route_s.tables(device, lbar, c, L // c)

    def fn():
        for t in route_s.conv(bu_re, bu_im, tb, c):
            ttnn.deallocate(t)

    ms = median_ms(fn, lambda: ttnn.synchronize_device(device))
    print("S time", c, shape, ms)
    dump(f"{OUT}/S{c}_time_B{B}_L{L}.json", {"B": B, "L": L, "c": c, "ms": ms, "timing": "synchronized wall, untraced"})


# ---------------------------------------------------------------- the dense part of one layer at paper


def test_dense_time(device):
    """B, C, gate, MLP, norms of one layer at paper (64 x 4096 tokens): bf16 activations and weights, HiFi2 with
    fp32 accumulation, HiFi4 for the B projection and the norms (CONTRACT §1 accuracy class)."""
    M = 64 * 4096
    hifi2 = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    hifi4 = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    bf = lambda *s: ttnn.from_torch(torch.randn(*s) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    u = bf(1, 1, M, 512)
    wB, wC, wG = bf(512, 1024), bf(1024, 512), bf(512, 512)
    w1, w2, w3 = bf(512, 1024), bf(512, 1024), bf(1024, 512)
    g512 = bf(1, 1, 1, 512)
    x = bf(1, 1, M, 1024)

    def fn():
        outs = []
        bu = ttnn.linear(u, wB, dtype=ttnn.float32, compute_kernel_config=hifi4)
        y = ttnn.linear(x, wC, compute_kernel_config=hifi2)
        y = ttnn.multiply(ttnn.sigmoid(ttnn.linear(y, wG, compute_kernel_config=hifi2)), y)
        y = ttnn.add(y, ttnn.multiply(u, g512))
        h = ttnn.rms_norm(y, weight=g512, epsilon=1e-6, compute_kernel_config=hifi4)
        a = ttnn.linear(h, w1, compute_kernel_config=hifi2, activation="silu")
        b = ttnn.linear(h, w2, compute_kernel_config=hifi2)
        z = ttnn.add(ttnn.linear(ttnn.multiply(a, b), w3, compute_kernel_config=hifi2), y)
        o = ttnn.add(u, ttnn.layer_norm(z, weight=g512, bias=g512, epsilon=1e-5, compute_kernel_config=hifi4))
        for t in (bu, y, h, a, b, z, o):
            ttnn.deallocate(t)

    ms = median_ms(fn, lambda: ttnn.synchronize_device(device))
    print("dense time", ms)
    dump(f"{OUT}/dense_time_paper.json", {"M": M, "ms": ms, "timing": "synchronized wall, untraced"})


# ---------------------------------------------------------------- route K (model-local generic_op, pre-approved)

from models.experimental.flowstate_forecast.ops.lti_scan import op as route_k  # noqa: E402


def run_k(device, Bu, lbar):
    lt = route_k.lbar_tiles(device, lbar)
    xr, xi = route_k.lti_scan(to_dev(device, Bu.real), to_dev(device, Bu.imag), lt)
    return torch.complex(ttnn.to_torch(xr).float(), ttnn.to_torch(xi).float())


@pytest.mark.parametrize("case", B0_CASES)
def test_route_k_accuracy(device, case):
    res = {}
    for layer in LAYERS:
        cs = case_inputs(case, layer)
        res[layer] = metrics(run_k(device, cs.Bu, cs.lbar), cs.ref, cs.truth)
        print("K", case, layer, res[layer], flush=True)
    dump(f"{OUT}/K_acc_{case}.json", res)


def test_route_k_control(device):
    """Positive control (D0 item 4): the recurrence with a bf16-rounded Lbar (bf16 state precision) must fail."""
    cs = case_inputs("b0_s1_L4096", 4)
    lb = torch.complex(cs.lbar.real.bfloat16().float(), cs.lbar.imag.bfloat16().float())
    m = metrics(run_k(device, cs.Bu, lb), cs.ref, cs.truth)
    print("K control bf16 Lbar", m)
    dump(f"{OUT}/K_control.json", m)
    assert m["rel_l2"] > 1e-3


@pytest.mark.parametrize("shape", [(64, 4096), (16, 16384)], ids=["paper", "long_context"])
def test_route_k_time(device, shape):
    B, L = shape
    g = torch.Generator().manual_seed(0)
    bu_re = to_dev(device, torch.randn(L, B, 512, generator=g))
    bu_im = to_dev(device, torch.randn(L, B, 512, generator=g))
    lt = route_k.lbar_tiles(device, torch.exp(torch.complex(-torch.rand(512) * 0.1, torch.rand(512))))

    def fn():
        for t in route_k.lti_scan(bu_re, bu_im, lt):
            ttnn.deallocate(t)

    ms = median_ms(fn, lambda: ttnn.synchronize_device(device))
    print("K time", shape, ms)
    dump(f"{OUT}/K_time_B{B}_L{L}.json", {"B": B, "L": L, "ms": ms, "timing": "synchronized wall, untraced (1 op)"})
