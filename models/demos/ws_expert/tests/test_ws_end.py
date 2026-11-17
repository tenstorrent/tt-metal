# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End pass (ws_end.py) on one chip: random result rows and gate weights, every token against torch; WS_WARM warm-up
+ WS_MEAS measured launches of one built program: wall time (synchronize to synchronize) and device time (the
cores' compute start to end stamps), and the DRAM bytes a second."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc
from models.demos.ws_expert.tests.test_ws_lane import first_visible
from models.demos.ws_expert.ws_end import build_end, end_cores, end_span_us, end_waits_k
from models.demos.ws_expert.ws_shared import SCfg, clusters

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def end_core_set(mesh, which):
    """The end pass's cores: "all"; "free20": the cores outside the shared expert's two cluster rectangles; "free26":
    those plus each rectangle's cores that neither compute nor feed x (ws_shared.clusters); "rows2": the grid's two
    last rows, every column; "rows2top": its first two rows."""
    cores = end_cores(mesh)
    if which == "all":
        return cores
    if which == "rows2":
        gy = max(y for _, y in cores) + 1
        return [c for c in cores if c[1] >= gy - 2]
    if which == "rows2top":
        return [c for c in cores if c[1] < 2]
    cl = clusters(mesh, SCfg(T=31250))
    rect = {c for r, _, _ in cl for c in r}
    out = [c for c in cores if c not in rect]
    if which == "free26":
        out += [c for r, act, fs in cl for c in r if c not in act and c not in fs]
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("T", [2048, 31250], ids=["t2048", "t31250"])
@pytest.mark.parametrize("nbuf", [4, 6, 2], ids=["nb4", "nb6", "nb2"])
def test_ws_end_chip(mesh_device, T, nbuf):
    K, H = 12, 3840
    S = K + 1
    torch.manual_seed(0)
    rows = torch.randn(T * S, H).to(torch.bfloat16)
    w = torch.zeros(T, 16)
    w[:, :K] = torch.rand(T, K)
    dram = ttnn.DRAM_MEMORY_CONFIG
    rows_t = ttnn.from_torch(
        rows, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram
    )
    w_t = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram)
    y_t = ttnn.empty([T, H], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram)
    fid = os.environ.get("WSE_FID", "HiFi4")
    rnoc = os.environ.get("WSE_READ_NOC", "0")
    cores = end_core_set(mesh_device, os.environ.get("WSE_CORES", "all"))
    mpd, io, st = build_end(
        mesh_device,
        T,
        K,
        H,
        rows_t,
        w_t,
        y_t,
        nbuf=nbuf,
        fidelity=getattr(ttnn.MathFidelity, fid),
        read_noc=rnoc,
        cores=cores,
    )
    clk = aiclk_mhz(first_visible())
    wall, dev = [], []
    for i in range(WARM + MEAS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        if i >= WARM:
            wall.append((time.perf_counter() - t0) * 1e3)
            dev.append(end_span_us(st, 0, clk) / 1e3)
    y = ttnn.to_torch(ttnn.get_device_tensors(y_t)[0]).float()
    wq = w[:, :K].to(torch.bfloat16).float()
    r = rows.reshape(T, S, H)
    ref = torch.empty(T, H)
    for a in range(0, T, 4096):
        b = min(T, a + 4096)
        ref[a:b] = torch.einsum("tk,tkh->th", wq[a:b], r[a:b, :K].float()) + r[a:b, K].float()
    p = pcc(y, ref)
    err = float((y - ref).abs().max() / ref.abs().max())
    gb = T * (S * H * 2 + 64 + H * 2) / 1e9
    dm, wm = statistics.median(dev), statistics.median(wall)
    logger.info(
        f"WSE end pass T {T} on {len(cores)} cores ({os.environ.get('WSE_CORES', 'all')}), {nbuf} tokens in flight a core, {fid}, reads on "
        f"NoC {rnoc}, {WARM} warm-up + {MEAS} measured launches, aiclk "
        f"{clk} MHz: device ms median {dm:.3f} (min {min(dev):.3f}), wall ms median {wm:.3f}; {gb:.2f} GB moved, "
        f"{gb / dm * 1e3:.0f} GB/s at device time; pcc {p:.6f}, max error {err:.2e} of max |y|; waits (K cycles, "
        f"core mean) {end_waits_k(st, 0)}"
    )
    assert p > 0.9999
