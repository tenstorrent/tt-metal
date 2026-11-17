# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The shared expert folded into the end pass (ws_fold.py), one chip: y = sum_k w_k routed row k + the shared row,
against the routed rows summed on the host (bf16 weights, fp32 sums) plus the shared expert's own rows from the
standalone op (ws_shared, column layout). WS_WARM warm-up + WS_MEAS measured launches: device time (the earliest
compute start to the latest compute end over the shared and end cores) and wall time."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc
from models.demos.ws_expert.tests.test_ws_lane import first_visible
from models.demos.ws_expert.ws_end import end_span_us, end_waits_k
from models.demos.ws_expert.ws_fold import build_fold, fold_span_us
from models.demos.ws_expert.ws_shared import SCfg, build_shared, shared_span_us, shared_weights

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("T", [2048, 31250], ids=["t2048", "t31250"])
def test_ws_fold_chip(mesh_device, T):
    K, H = 12, 3840
    cfg = SCfg(T=T)
    fid = getattr(ttnn.MathFidelity, os.environ.get("WSF_FID", "HiFi2"))
    top = os.environ.get("WSF_ROWS", "top") == "top"
    I = cfg.IS
    TX = -(-(T + 1) // 32) * 32
    torch.manual_seed(0)
    x = torch.zeros(TX, H, dtype=torch.bfloat16)
    x[:T] = torch.randn(T, H).to(torch.bfloat16)
    w1 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
    w3 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(I, H) / I**0.5).to(torch.bfloat16)
    rows = torch.randn(T * K + 16, H).to(torch.bfloat16)
    w = torch.zeros(TX, 16)
    w[:T, :K] = torch.rand(T, K)
    # (checks: WSF_W0, the routed weights 0 - y is the shared rows alone; WSF_NOSHARED, the shared rows' weight 0)
    if os.environ.get("WSF_W0", "0") == "1":
        w.zero_()
    no_shared = os.environ.get("WSF_NOSHARED", "0") == "1"
    dram = ttnn.DRAM_MEMORY_CONFIG
    rm = ttnn.ROW_MAJOR_LAYOUT
    put = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=rm, device=mesh_device, memory_config=dram)
    x_t, rows_t, w_t = put(x, ttnn.bfloat16), put(rows, ttnn.bfloat16), put(w, ttnn.float32)
    w13_t, w2_t = shared_weights(mesh_device, cfg, w1, w3, w2)
    y_t = ttnn.empty([T, H], dtype=ttnn.bfloat16, layout=rm, device=mesh_device, memory_config=dram)
    fence_t = ttnn.empty([16, 16], dtype=ttnn.uint32, layout=rm, device=mesh_device, memory_config=dram)
    clk = aiclk_mhz(first_visible())
    # the reference's shared rows: the standalone op (column layout) into row t S + K of a 13-row buffer
    S = K + 1
    res13 = ttnn.zeros([T * S + 16, H], dtype=ttnn.bfloat16, layout=rm, device=mesh_device, memory_config=dram)
    smpd, sio, _, _ = build_shared(mesh_device, cfg, x_t, w13_t, w2_t, res13)
    ttnn.generic_op(sio, smpd)
    ttnn.synchronize_device(mesh_device)
    shared = ttnn.to_torch(ttnn.get_device_tensors(res13)[0])[: T * S].reshape(T, S, H)[:, K].float()
    del smpd, sio, res13
    mpd, io, sts = build_fold(
        mesh_device, cfg, K, x_t, w13_t, w2_t, rows_t, w_t, y_t, fence_t, top=top, fidelity=fid, no_shared=no_shared
    )
    wall, dev, spans = [], [], []
    for i in range(WARM + MEAS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        if i >= WARM:
            wall.append((time.perf_counter() - t0) * 1e3)
            dev.append(fold_span_us(sts, 0, clk) / 1e3)
            spans.append((shared_span_us(sts[:2], 0, clk) / 1e3, end_span_us(sts[2], 0, clk) / 1e3))
    y = ttnn.to_torch(ttnn.get_device_tensors(y_t)[0]).float()
    wq = w[:T, :K].to(torch.bfloat16).float()
    r = rows[: T * K].reshape(T, K, H)
    ref = torch.empty(T, H)
    for a in range(0, T, 4096):
        b = min(T, a + 4096)
        ref[a:b] = torch.einsum("tk,tkh->th", wq[a:b], r[a:b].float()) + (0 if no_shared else shared[a:b])
    p = pcc(y, ref)
    tok_err = (y - ref).abs().amax(1) / ref.abs().amax().clamp_min(1e-6)
    bad = torch.nonzero(tok_err > 2e-2).flatten()
    col_err = (y - ref).abs().amax(0).reshape(-1, 32).amax(1) / ref.abs().amax().clamp_min(1e-6)
    logger.info(
        f"WSF check T {T} w0 {bool(w.abs().sum() == 0)} no_shared {no_shared}: {len(bad)} of {T} tokens off; first "
        f"{bad[:12].tolist()} (in their row tile: {(bad[:12] % 32).tolist()}); 32-column blocks off "
        f"{torch.nonzero(col_err > 2e-2).flatten()[:24].tolist()}"
    )
    err = float((y - ref).abs().max() / ref.abs().max())
    gb = T * (K * H * 2 + 64 + H * 2) / 1e9
    dm = statistics.median(dev)
    logger.info(
        f"WSF fold T {T}, end rows {'top' if top else 'bottom'}, end MAC {fid}, {WARM} warm-up + {MEAS} measured "
        f"launches, aiclk {clk} MHz: device ms median {dm:.3f} (min {min(dev):.3f}; shared cores "
        f"{statistics.median(s for s, _ in spans):.3f}, end cores {statistics.median(e for _, e in spans):.3f}), wall "
        f"ms median {statistics.median(wall):.3f}; {gb:.2f} GB through DRAM, {gb / dm * 1e3:.0f} GB/s; pcc {p:.6f}, "
        f"max error {err:.2e} of max |y|; end waits (K cycles, core mean) {end_waits_k(sts[2], 0)}, slot flag waits "
        f"{float((ttnn.to_torch(ttnn.get_device_tensors(sts[2])[0]).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 512)[:, 7].double().mean()) / 1000:.1f}"
    )
    assert p > 0.9999
