# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared expert (ws_shared.py) on one chip: random x and weights, the result rows t S + KSH against torch (the
device's quantization: bfp4 / bfp8 weights, bfp8 x rows, bf16 h); every token at T 2048, WSS_TOKENS sampled ones at
T 31250. WS_WARM warm-up + WS_MEAS measured launches (the program built again each launch: fresh flag words): wall
time of the launch and device time (the cores' compute start to end), util against 2 T H I 3 FLOP."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize
from models.demos.ws_expert.tests.test_ws_lane import first_visible
from models.demos.ws_expert.ws_shared import SCfg, build_shared, shared_span_us, shared_waits_k, shared_weights

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("T", [2048, 31250], ids=["t2048", "t31250"])
def test_ws_shared_chip(mesh_device, T):
    prec = os.environ.get("WSS_PREC", "bfp4_lofi")
    w_dt, fid = {"bfp4_lofi": (ttnn.bfloat4_b, "LoFi"), "bfp8_hifi2": (ttnn.bfloat8_b, "HiFi2")}[prec]
    nf, xrm, xst = (int(os.environ.get(k, d)) for k, d in (("WSS_NF", "1"), ("WSS_XRM", "1"), ("WSS_XST", "2")))
    bk, nx = int(os.environ.get("WSS_BK", "24")), int(os.environ.get("WSS_NX", "9"))
    h_dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("WSS_HDT", "bfp8")]
    cfg = SCfg(
        T=T, w_dt=w_dt, fidelity=getattr(ttnn.MathFidelity, fid), NF=nf, XRM=xrm, XST=xst, BK=bk, NX=nx, h_dt=h_dt
    )
    H, I, S = cfg.H, cfg.IS, cfg.S
    TX = -(-(T + 1) // 32) * 32
    torch.manual_seed(0)
    x = torch.zeros(TX, H, dtype=torch.bfloat16)
    x[:T] = torch.randn(T, H).to(torch.bfloat16)
    w1 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
    w3 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(I, H) / I**0.5).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram)
    w13_t, w2_t = shared_weights(mesh_device, cfg, w1, w3, w2)
    res_t = ttnn.zeros(
        [T * S + 16, H], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram
    )
    clk = aiclk_mhz(first_visible())
    wall, dev = [], []
    for i in range(WARM + MEAS):
        mpd, io, st, active = build_shared(mesh_device, cfg, x_t, w13_t, w2_t, res_t)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        if i >= WARM:
            wall.append((time.perf_counter() - t0) * 1e3)
            dev.append(shared_span_us(st, 0, clk) / 1e3)
        if i == WARM + MEAS - 1:
            waits = shared_waits_k(st, 0)
        del mpd, io, st
    ntok = int(os.environ.get("WSS_TOKENS", 0 if T <= 4096 else 256))
    toks = torch.arange(T) if ntok <= 0 or ntok >= T else torch.linspace(0, T - 1, ntok).round().long()
    res = ttnn.to_torch(ttnn.get_device_tensors(res_t)[0]).float()[: T * S].reshape(T, S, H)
    got = res[toks, cfg.KSH]
    X = quantize(x[toks].float(), cfg.x_dt)
    q = lambda w: quantize(w.float(), cfg.w_dt)
    h = quantize(torch.nn.functional.silu(X @ q(w1)) * (X @ q(w3)), cfg.h_dt)
    ref = h @ q(w2)
    p = pcc(got, ref)
    others = float(res[toks][:, : cfg.KSH].abs().max())
    fl = 2 * T * H * I * 3
    dm, wm = statistics.median(dev), statistics.median(wall)
    sol = 110 * 4096 * clk * 1e6 / (1 if fid == "LoFi" else 2)
    logger.info(
        f"WSS shared expert T {T}, {prec}, {len(active)} cores ({cfg.NCL} clusters x {cfg.GC}), {nf} x feeders a "
        f"cluster ({xrm} / {xst} staged row tiles), BK {bk} NX {nx} h {h_dt}, {WARM} warm-up + "
        f"{MEAS} measured launches, aiclk {clk} MHz: device ms median {dm:.3f} (min {min(dev):.3f}), wall ms median "
        f"{wm:.3f}; {fl / 1e12:.2f} TFLOP: util {100 * fl / sol / (dm / 1e3):.1f}% of the 110-core {fid} SOL "
        f"({fl / sol * 1e3:.2f} ms), {100 * fl * 110 / len(active) / sol / (dm / 1e3):.1f}% of the {len(active)} "
        f"cores; pcc {p:.6f} over {len(toks)} tokens, max error {float((got - ref).abs().max() / ref.abs().max()):.2e}"
        f"; other result rows max |.| {others}; waits (K cycles, core mean) {waits}"
    )
    assert p > 0.999 and others == 0.0
