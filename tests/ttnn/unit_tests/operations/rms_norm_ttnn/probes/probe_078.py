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

# name -> (shape, shard, memory_layout, mode, fp32_dest)
CASE = os.environ.get("RMS_CASE", "focus")
CASES = {
    # THE FOCUS SHAPE
    "focus": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False),
    # runner-up / guard geometries
    "blk7168_gbr": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False),
    "w5120_gbr": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True),
    "w2304": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False),
    "w1024": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False),
    "w5120": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False),
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
    shape, shard, ml, mode, fp32_dest = CASES[name]
    W = shape[-1]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.TILE_LAYOUT
    mc = shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
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
    return agg, spans, target_run


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

    agg, spans, target_run = parse_zones()
    FREQ = 1350.0  # MHz -> cycles to ns: /1.35
    print(f"RESULT ---- per-stage zones (run host id {target_run}) ----")
    print(f"RESULT {'zone':26s} {'risc':8s} {'n':>5s} {'tot_ns':>10s} {'mean_ns':>9s} {'max_ns':>9s}")
    rowsout = []
    for (zone, risc), vals in agg.items():
        # per-core mean across all cores/occurrences; report per-core totals
        ncore = len({1})
        tot = sum(vals) / FREQ
        rowsout.append((sum(vals), zone, risc, len(vals), tot, statistics.mean(vals) / FREQ, max(vals) / FREQ))
    for _, zone, risc, n, tot, mean, mx in sorted(rowsout, reverse=True):
        print(f"RESULT {zone:26s} {risc:8s} {n:5d} {tot:10.0f} {mean:9.1f} {mx:9.1f}")
    print("RESULT ---- per-RISC kernel span (max over cores), ns ----")
    byrisc = collections.defaultdict(list)
    for (cx, cy, risc), (lo, hi) in spans.items():
        if lo <= hi:
            byrisc[risc].append((hi - lo) / FREQ)
    for risc, v in sorted(byrisc.items()):
        print(f"RESULT span {risc:8s} ncores={len(v):3d} max={max(v):9.1f} mean={statistics.mean(v):9.1f}")


main()
