"""Perf round 1, Step 1 -- per-stage device-ns breakdown on the FOCUS SHAPE.

Focus shape (ranked worst of the 19-case `perf` group at ratio 0.997):
    (1,1,32,7168) WIDTH_SHARDED shard [32,256] grid (7,4) = 28 cores,
    gamma (bf16 TILE weight), dtype bf16, fp32_dest_acc_en=False, HiFi2, TILE.

Runs the op with RMS_STAGE_ZONES=1 and aggregates every `MaybeDeviceZoneScope`
occurrence out of generated/profiler/.logs/profile_log_device.csv, per
(zone, RISC), reporting count / total / mean / max in CYCLES and ns, plus the
per-RISC kernel span so we can see whether the zones cover the whole kernel.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_TRACE_BLOCKING"] = "1"
os.environ["RMS_STAGE_ZONES"] = "1"

import csv
import collections
import shutil
import statistics
import sys
from pathlib import Path

import torch
import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import shard_config

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
LOGDIR = Path("generated/profiler/.logs")
CSV = LOGDIR / "profile_log_device.csv"

# name -> (shape, memory_layout|None, shard|None, mode, fp32_dest)
CASE = os.environ.get("RMS_CASE", "focus")
_INT = None
CASES = {
    # ROUND 2 FOCUS: worst measured/achievable ratio of the perf group (0.908)
    "focus": ((1, 1, 8192, 2304), _INT, None, "gamma", False),
    "focus_nog": ((1, 1, 8192, 2304), _INT, None, "none", False),
    "run1024": ((1, 1, 8192, 1024), _INT, None, "gamma", False),
    "run1024_nog": ((1, 1, 8192, 1024), _INT, None, "none", False),
    "stream7168": ((1, 1, 8192, 7168), _INT, None, "gamma_bias_residual", True),
    "w7168": ((1, 1, 8192, 7168), _INT, None, "gamma", False),
    "w5120gbr": ((1, 1, 8192, 5120), _INT, None, "gamma_bias_residual", True),
}


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def build(device, name):
    shape, ml, shard, mode, fp32_dest = CASES[name]
    W = shape[-1]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.TILE_LAYOUT
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if ml is None
        else shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    )
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)

    if "gamma" in mode:
        t, v = _vec(1)
        kwargs["weight"] = v
        ref["weight"] = t.float()
    if "bias" in mode:
        t, v = _vec(2)
        kwargs["bias"] = v
        ref["bias"] = t.float()
    if "residual" in mode:
        torch.manual_seed(3)
        tr = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            tr, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc
        )
        ref["residual_input_tensor"] = tr.float()
    expected = torch_rms_norm_ttnn(
        ref["input_tensor"],
        epsilon=1e-12,
        weight=ref.get("weight"),
        bias=ref.get("bias"),
        residual_input_tensor=ref.get("residual_input_tensor"),
    )
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def parse_zones():
    """Aggregate ZONE_START/ZONE_END pairs per (zone, risc). Returns dict + spans."""
    rows = []
    with CSV.open() as fh:
        # first line is the ARCH header
        first = fh.readline()
        rdr = csv.reader(fh)
        header = next(rdr)
        header = [h.strip() for h in header]
        idx = {h: i for i, h in enumerate(header)}
        for r in rdr:
            if len(r) < len(header):
                continue
            rows.append(r)
    ci = {
        k: idx[k]
        for k in (
            "core_x",
            "core_y",
            "RISC processor type",
            "time[cycles since reset]",
            "zone name",
            "type",
            "run host ID",
        )
    }
    # take the LAST run host id present (most recent program)
    runs = sorted({int(r[ci["run host ID"]]) for r in rows})
    target_run = runs[-1]
    open_stack = collections.defaultdict(list)
    agg = collections.defaultdict(list)
    zcores = collections.defaultdict(set)
    spans = collections.defaultdict(lambda: [float("inf"), -float("inf")])
    for r in rows:
        if int(r[ci["run host ID"]]) != target_run:
            continue
        key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
        zone = r[ci["zone name"]].strip()
        typ = r[ci["type"]].strip()
        t = int(r[ci["time[cycles since reset]"]])
        spans[key][0] = min(spans[key][0], t)
        spans[key][1] = max(spans[key][1], t)
        if typ == "ZONE_START":
            open_stack[(key, zone)].append(t)
        elif typ == "ZONE_END":
            st = open_stack[(key, zone)]
            if st:
                agg[(zone, r[ci["RISC processor type"]])].append(t - st.pop())
                zcores[(zone, r[ci["RISC processor type"]])].add(key)
    return agg, spans, target_run, zcores


def main():
    name = CASE
    if CSV.exists():
        CSV.unlink()
    device = ttnn.open_device(device_id=0)
    try:
        run, expected = build(device, name)
        out = run()
        got = ttnn.to_torch(out)
        p = pcc(got, expected)
        del out, got
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        print(f"RESULT case={name} first_run_ns={ns} pcc={p:.6f}")
        # one more fresh measurement for the zone capture
        run()
        ttnn.synchronize_device(device)
        ns2 = _read_kernel_ns(device)
        print(f"RESULT case={name} zone_capture_ns={ns2}")
    finally:
        ttnn.close_device(device)

    agg, spans, target_run, zcores = parse_zones()
    FREQ = 1.35  # cycles -> ns at 1350 MHz
    print(f"RESULT ---- per-stage zones (run host id {target_run}) ----")
    print(
        f"RESULT {'zone':26s} {'risc':8s} {'ncore':>5s} {'n/core':>6s} {'ns/core':>10s} {'mean_ns':>9s} {'max_ns':>9s}"
    )
    rowsout = []
    for (zone, risc), vals in agg.items():
        nc = max(1, len(zcores[(zone, risc)]))
        per_core = sum(vals) / FREQ / nc
        rowsout.append((per_core, zone, risc, nc, len(vals) / nc, statistics.mean(vals) / FREQ, max(vals) / FREQ))
    for per_core, zone, risc, nc, npc, mean, mx in sorted(rowsout, reverse=True):
        print(f"RESULT {zone:26s} {risc:8s} {nc:5d} {npc:6.1f} {per_core:10.0f} {mean:9.1f} {mx:9.1f}")
    print("RESULT ---- per-RISC kernel span (max over cores), ns ----")
    byrisc = collections.defaultdict(list)
    for (cx, cy, risc), (lo, hi) in spans.items():
        if lo <= hi:
            byrisc[risc].append((hi - lo) / FREQ)
    for risc, v in sorted(byrisc.items()):
        print(f"RESULT span {risc:8s} ncores={len(v):3d} max={max(v):9.1f} mean={statistics.mean(v):9.1f}")


main()
