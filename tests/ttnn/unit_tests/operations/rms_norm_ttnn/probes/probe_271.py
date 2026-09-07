"""Perf 3, Step 1 — per-stage device zones on the focus shape.

Occupancy caveat (references/device-zone-scope-attribution.md): `reader_read_x`,
`writer_write` and `compute_square` wrap their OWN cb_wait_front / cb_reserve_back,
so their numbers are WAIT + WORK -- occupancy, not payload.  The RANKING is taken
from the cumulative peel (peel.sh), never from these; the zones are here to say WHICH
thread is holding the wall and to bound the per-block fixed costs.
"""
import collections
import csv
import os
import statistics
from pathlib import Path

os.environ["RMS_STAGE_ZONES"] = "1"
for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)
import ttnn  # noqa: E402
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn  # noqa: E402

CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35  # GHz -> cycles to ns


def parse_zones():
    rows = []
    with CSVP.open() as fh:
        fh.readline()
        rdr = csv.reader(fh)
        header = [h.strip() for h in next(rdr)]
        idx = {h: i for i, h in enumerate(header)}
        for r in rdr:
            if len(r) >= len(header):
                rows.append(r)
    ci = {
        k: idx[k]
        for k in ("core_x", "core_y", "RISC processor type", "time[cycles since reset]", "zone name", "type", "run host ID")
    }
    target = sorted({int(r[ci["run host ID"]]) for r in rows})[-1]
    open_stack = collections.defaultdict(list)
    per_core = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.defaultdict(int)
    span = collections.defaultdict(lambda: [None, None])
    for r in rows:
        if int(r[ci["run host ID"]]) != target:
            continue
        key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
        zone = r[ci["zone name"]].strip()
        t = int(r[ci["time[cycles since reset]"]])
        lo, hi = span[key]
        span[key] = [t if lo is None else min(lo, t), t if hi is None else max(hi, t)]
        if r[ci["type"]].strip() == "ZONE_START":
            open_stack[(key, zone)].append(t)
        elif r[ci["type"]].strip() == "ZONE_END" and open_stack[(key, zone)]:
            dt = t - open_stack[(key, zone)].pop()
            per_core[(zone, r[ci["RISC processor type"]])][key] += dt
            counts[(zone, r[ci["RISC processor type"]])] += 1
    return per_core, counts, target, span


def main():
    import torch

    shape = (1, 1, 8192, 1024)
    W = shape[-1]
    if CSVP.exists():
        CSVP.unlink()
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        torch.manual_seed(1)
        tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        gm = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = rms_norm_ttnn(x, epsilon=1e-12, compute_kernel_config=cfg, weight=gm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)
    per_core, counts, target, span = parse_zones()
    print(f"RESULT ---- zones {shape} (run {target}) ----")
    print(f"RESULT {'zone':26s} {'risc':8s} {'occ/core ns':>12s} {'max/core':>10s} {'n/core':>8s} {'cores':>6s}")
    out = []
    for k, cores in per_core.items():
        vals = list(cores.values())
        out.append((statistics.mean(vals), k[0], k[1], max(vals), counts[k] / len(vals), len(vals)))
    for mean, z, ri, mx, n, nc in sorted(out, reverse=True):
        print(f"RESULT {z:26s} {ri:8s} {mean / FREQ:12.1f} {mx / FREQ:10.1f} {n:8.2f} {nc:6d}")
    print("RESULT ---- per-RISC total marker span (kernel coverage check) ----")
    byrisc = collections.defaultdict(list)
    for (cx, cy, ri), (lo, hi) in span.items():
        byrisc[ri].append((hi - lo) / FREQ)
    for ri, v in sorted(byrisc.items()):
        print(f"RESULT span {ri:8s} mean={statistics.mean(v):10.1f} max={max(v):10.1f} n={len(v)}")


main()
