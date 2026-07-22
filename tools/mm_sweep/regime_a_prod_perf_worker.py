#!/usr/bin/env python3
# Per-shape perf worker: measure the PRODUCTION ttnn op (ttnn.experimental.regime_a_matmul, config=None ->
# auto-picker -> chain reduction) kernel wall via the device profiler. One shape per process (fresh device +
# fresh CSV) mirrors the proven golden-parity harness. Prints "WALL_US=<median-over-runs kernel us>".
# argv: M K N iters
import sys, os, csv, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9  # BH
M, K, N, iters = (int(x) for x in sys.argv[1:5])


def parse_wall_us():
    # total kernel = max across ALL (core,-KERNEL zone) per run; median over runs 1.. (drop warmup run 0).
    rows = list(csv.reader(open(CSV)))
    ev = {}
    for row in rows[2:]:
        if len(row) < 12:
            continue
        z = row[10].strip()
        if not z.endswith("-KERNEL"):
            continue
        ev.setdefault(((row[1], row[2], row[3]), z), []).append((row[11].strip(), int(row[5])))
    dur = {}
    for k, lst in ev.items():
        ds, st = [], None
        for t, c in lst:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        dur[k] = ds
    nruns = min((len(v) for v in dur.values()), default=0)
    if nruns < 2:
        return None
    walls = [max(v[i] for v in dur.values()) for i in range(1, nruns)]
    return statistics.median(walls) / FREQ * 1e6


try:
    os.remove(CSV)
except OSError:
    pass

dev = ttnn.open_device(device_id=0)
try:
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
    b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
    out = None
    for _ in range(iters + 1):  # warmup (run 0, dropped) + iters timed
        out = ttnn.experimental.regime_a_matmul(a, b)  # config=None -> production default (chain)
    ttnn.synchronize_device(dev)
    _ = ttnn.to_torch(ttnn.from_device(out))  # force completion
finally:
    ttnn.close_device(dev)  # flushes the device-profiler CSV

w = parse_wall_us()
print(f"WALL_US={w}")
