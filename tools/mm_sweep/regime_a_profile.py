#!/usr/bin/env python3
# Per-RISC critical-path profiler for ttnn.experimental.regime_a_matmul (one config per subprocess).
# Complements regime_a_bench.py: instead of a single max-over-cores kernel time, it breaks the steady
# -state iteration down per RISC role so we can say WHICH engine bounds the shape.
#
# RISC roles in this op:
#   - in1 reader           -> the DRAM-sharded reader==consumer kernel (BRISC or NCRISC data-movement).
#   - in0 ring/reduce writer -> the in0 ring all-gather + split-K linear-reduction writer (the other DM RISC).
#   - compute (TRISC)      -> unpack/math/pack (the profiler reports one TRISC-KERNEL zone per core).
# The zone name (CSV col 10) carries the RISC tag ("<RISC>-KERNEL"); col 11 = ZONE_START/ZONE_END; col 5 = cycle.
#
# Usage:  python3 regime_a_profile.py --run M K N Ns Pk Sm kb nsb     (worker; prints RISCJSON line)
#         python3 regime_a_profile.py M K N Ns Pk Sm kb nsb           (driver; launches worker, reports)
import csv, json, os, subprocess, sys, statistics
from collections import defaultdict

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN_CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return cdiv(x, y) * y


def parse_per_risc():
    """Return {risc_tag: {'per_core_cycles': [median-over-runs busy cycles for each core]}} and the
    per-run wall (max over all (core,risc) zones). Drops run 0 (warmup)."""
    rows = list(csv.reader(open(BIN_CSV)))
    # (x,y,zone) -> list[(marker, cycle)]
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[1], row[2], row[3], row[10].strip())].append((row[11].strip(), int(row[5])))
    # per (core,zone): list of per-run durations
    dur = {}
    for k, l in ev.items():
        ds, st = [], None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        if ds:
            dur[k] = ds[1:] if len(ds) > 1 else ds  # drop warmup run 0
    # group by RISC tag (zone name without the core), collect per-core median-over-runs
    by_risc = defaultdict(list)
    nruns = min((len(v) for v in dur.values()), default=0)
    for (x, y, z, zone), ds in dur.items():
        tag = zone[: -len("-KERNEL")]
        by_risc[tag].append(statistics.median(ds))
    # per-run wall = max over all zones at each run index
    wall = [max(v[i] for v in dur.values()) for i in range(nruns)] if nruns else []
    return by_risc, wall


def worker(M, K, N, Ns, Pk, Sm, kb, nsb):
    import torch
    import ttnn
    from models.common.utility_functions import comp_pcc

    try:
        os.remove(BIN_CSV)
    except OSError:
        pass
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        ref = (t0.float() @ t1.float())[0, 0]
        in0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        in1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
        out = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
        got = ttnn.to_torch(ttnn.from_device(out))[0, 0].float()
        ok, pcc = comp_pcc(ref, got, 0.999)
        for _ in range(8):
            o = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
            ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)
    by_risc, wall = parse_per_risc()
    summary = {
        tag: {
            "n_cores": len(cyc),
            "max_us": max(cyc) / FREQ * 1e6,
            "med_us": statistics.median(cyc) / FREQ * 1e6,
            "min_us": min(cyc) / FREQ * 1e6,
        }
        for tag, cyc in sorted(by_risc.items())
    }
    wall_us = statistics.median(wall) / FREQ * 1e6 if wall else None
    print("RISCJSON " + json.dumps({"pcc": float(pcc), "wall_us": wall_us, "risc": summary}))


def driver(M, K, N, Ns, Pk, Sm, kb, nsb):
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole", PYTHONPATH=ROOT)
    args = [str(a) for a in (M, K, N, Ns, Pk, Sm, kb, nsb)]
    cmd = ["timeout", "-s", "TERM", "200", sys.executable, __file__, "--run"] + args
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith("RISCJSON "):
            d = json.loads(line[9:])
            Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
            Nband = cdiv(Nt, 8)
            Ktl = rup(cdiv(Kt, Pk), kb * 8)
            Mblk = cdiv(Mt, Sm)
            Nown = cdiv(Nband, Ns)
            print(f"\n=== {M}x{K}x{N}  cfg (Ns,Pk,Sm,kb,nsb)=({Ns},{Pk},{Sm},{kb},{nsb})  cores={8*Pk*Ns*Sm} ===")
            print(f"geometry: Mt={Mt} Kt={Kt} Nt={Nt} | Ktl(K-slice cap)={Ktl} Mblk={Mblk} Nown={Nown}")
            print(f"wall (median kernel us) = {d['wall_us']:.2f}   pcc={d['pcc']:.5f}")
            print(f"{'RISC':<16}{'cores':>7}{'max_us':>10}{'med_us':>10}{'min_us':>10}")
            for tag, s in d["risc"].items():
                print(f"{tag:<16}{s['n_cores']:>7}{s['max_us']:>10.2f}{s['med_us']:>10.2f}{s['min_us']:>10.2f}")
            return d
    print("NO RISCJSON; stderr tail:\n" + r.stderr[-800:])
    return None


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        worker(*[int(x) for x in sys.argv[2:10]])
    else:
        driver(*[int(x) for x in sys.argv[1:9]])
