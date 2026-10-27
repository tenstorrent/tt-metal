# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmark (D2): fused sLSTM kernel vs the op-level loop, B = 1024 rows, T = 285 steps.

Same WxT (block input through W_in, rows tiled with noise) for both. Reports us per step (warm, wall clock
with device sync) and the final-step PCC of y, c, n, m. Gate: <= 40 us per step, PCC >= 0.9999.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights, pcc

pytestmark = [
    pytest.mark.skipif(not have_goldens(), reason="goldens not generated"),
    pytest.mark.use_module_device({"l1_small_size": 16384}),
]

GATE_US = 40.0
FLOOR = 0.9999
B, T = 1024, 285


def _wx(device, model, b: int):
    import ttnn

    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.common import upload

    x = golden("slstm_short")[f"block{b}.x"] / host_weights().blocks[b].w_norm_lstm
    rows, steps, _ = x.shape
    gen = torch.Generator().manual_seed(b)
    xt = x.repeat(-(-B // rows), -(-T // steps), 1)[:B, :T]
    xt[rows:] += 0.3 * torch.randn(xt[rows:].shape, generator=gen)
    cell = model.weights.blocks[b].cell
    xT = upload(device, xt.transpose(0, 1).reshape(T, 1, B, 512))
    WxT = ttnn.add(layers.linear(xT, cell.w_in, model.precision, dtype=ttnn.float32), cell.bias)
    return WxT, cell


def _time(device, fn, reps=2):
    import ttnn

    out = fn()  # warm-up / compile
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return out, best * 1e6 / T


def test_slstm_step_bench(device):
    import ttnn

    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    WxT, cell = _wx(device, model, 1)

    (ref_ys, (rc, rn, rm)), loop_us = _time(
        device, lambda: layers.slstm_recurrence(device, WxT, cell.r, T, model.consts), reps=1
    )
    g = int(os.environ.get("SLSTM_BENCH_GABL", "0"))  # try a gate variant (e.g. 8: no logsig exp tail)
    (ys, cs, ns, ms), fused_us = _time(device, lambda: slstm_steps(WxT, cell.r, T, gate_ablate=g))

    ref = [ttnn.to_torch(t).float().reshape(B, 512) for t in (ref_ys[-1], rc, rn, rm)]
    got = [ttnn.to_torch(ys).float()[-1].reshape(B, 512)] + [
        ttnn.to_torch(t).float().reshape(B, 512) for t in (cs, ns, ms)
    ]
    p = {nm: pcc(a, g) for nm, a, g in zip("ycnm", ref, got)}
    print(
        f"\nBENCH gabl={g} B={B} T={T}: loop {loop_us:.1f} us/step, fused {fused_us:.1f} us/step, "
        f"speedup {loop_us / fused_us:.1f}x; last-step pcc " + " ".join(f"{k}={v:.7f}" for k, v in p.items())
    )
    model.release()
    assert min(p.values()) >= FLOOR
    assert fused_us <= GATE_US


def test_slstm_step_ablate(device):
    """Time attribution only: fused us/step with parts of the step removed (results are wrong by design)."""
    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    WxT, cell = _wx(device, model, 1)
    names = {
        0: "full",
        7: "skeleton",
        256: "no DRAM y",
        128 | 512: "no peer exchange",
        1024: "no c/n/m ring",
        135 | 256 | 512 | 1024: "bare skeleton",
        128 | 256 | 512 | 1024: "full math, no I/O",
    }
    for a, nm in names.items():
        _, us = _time(device, lambda: slstm_steps(WxT, cell.r, T, ablate=a))
        print(f"\nABLATE {a} ({nm}): {us:.1f} us/step")
    model.release()


def test_slstm_step_gate_ablate(device):
    """Time attribution inside the one-pass SFPI gate chain (results are wrong by design)."""
    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    WxT, cell = _wx(device, model, 1)
    names = {
        0: "full",
        1: "exp -> x",
        2: "recip no Newton",
        4: "logsig -> f",
        8: "no logsig tail exp",
        15: "all",
        16: "no gates (ablate 2)",
    }
    for g, nm in names.items():
        kw = dict(ablate=2) if g == 16 else dict(gate_ablate=g)
        _, us = _time(device, lambda: slstm_steps(WxT, cell.r, T, **kw))
        print(f"\nGABL {g} ({nm}): {us:.1f} us/step")
    model.release()


def test_slstm_step_compute_ablate(device):
    """Split the compute-thread floor: on top of the bare skeleton, drop Wx copy / y packs / per-item inits."""
    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    WxT, cell = _wx(device, model, 1)
    bare = 135 | 256 | 512 | 1024
    names = {
        (0, 0): "full",
        (bare, 0): "bare skeleton",
        (bare, 1): "bare, no Wx copy",
        (bare, 2): "bare, no y packs",
        (bare, 4): "bare, inits hoisted",
        (bare, 7): "bare, all three",
        (0, 4): "full, inits hoisted",
    }
    pick = os.environ.get("SLSTM_CABL_CASE")  # e.g. "0,0": run one case per job (job 795 hung)
    if pick:
        a0, c0 = (int(v) for v in pick.split(","))
        names = {(a0, c0): names.get((a0, c0), "custom")}
    for (a, c), nm in names.items():
        _, us = _time(device, lambda: slstm_steps(WxT, cell.r, T, ablate=a, compute_ablate=c))
        print(f"\nCABL a={a} c={c} ({nm}): {us:.1f} us/step")
    model.release()


def test_slstm_step_profile(device):
    """One launch with device-profiler zones on steps 100-103 (run with TT_METAL_DEVICE_PROFILER=1)."""
    import csv
    from collections import defaultdict

    import ttnn

    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    WxT, cell = _wx(device, model, 1)
    slstm_steps(WxT, cell.r, T)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    path = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
    with open(path) as f:
        mhz = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    open_, dur = {}, defaultdict(list)
    for r in rows:
        name = r["zone name"]
        if name not in ("C-ITEM", "C-READY", "R-PEER"):
            continue
        key = (r["core_x"], r["core_y"], r["RISC processor type"], name)
        ts = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            open_[key] = ts
        elif key in open_:
            dur[key].append((ts - open_.pop(key)) / mhz)
    per = defaultdict(list)  # zone -> per-core total us over 4 steps
    for (x, y, risc, name), d in dur.items():
        per[name].append((sum(d) / 4, len(d) / 4, x, y))
    for name, v in per.items():
        v.sort()
        print(
            f"\nPROFILE {name}: cores {len(v)}, us/step per core min {v[0][0]:.2f} median {v[len(v)//2][0]:.2f} "
            f"max {v[-1][0]:.2f} (core {v[-1][2]},{v[-1][3]}, {v[-1][1]:.0f} zones/step)"
        )
    model.release()


def test_slstm_step_host(device):
    """Host vs device split of one fused call: enqueue time (no sync) and total wall time, warm, best of 3."""
    import ttnn

    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    WxT, cell = _wx(device, model, 1)
    slstm_steps(WxT, cell.r, T)
    ttnn.synchronize_device(device)
    enq, wall = float("inf"), float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        slstm_steps(WxT, cell.r, T)
        t1 = time.perf_counter()
        ttnn.synchronize_device(device)
        enq, wall = min(enq, t1 - t0), min(wall, time.perf_counter() - t0)
    print(
        f"\nHOST B={B} T={T}: enqueue {enq * 1e3:.2f} ms ({enq * 1e6 / T:.1f} us/step), "
        f"wall {wall * 1e3:.2f} ms ({wall * 1e6 / T:.1f} us/step)"
    )
    model.release()
