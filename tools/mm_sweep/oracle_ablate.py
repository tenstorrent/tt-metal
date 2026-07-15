#!/usr/bin/env python3
# Controlled oracle ablations (Part 5) + per-core/per-RISC profiling (Part 6) for the retained C++
# prototype `test_regime_a_mm` (the DIAGNOSTIC oracle, NOT the shipped TTNN op). Runs the binary with a
# (Ns,Pk,Sm,kb,nsb) geometry matching the TTNN winner + an ablation flag set, parses the device-profiler
# CSV into per-(core,RISC) KERNEL durations, and reports the wall (max over cores) + per-RISC distribution.
#
# Ablations are critical-path COUNTERFACTUALS, not additive (stages overlap):
#   full ~= skipin1                    -> in1 delivery not limiting
#   full improves w/ skipin0|skipfwd   -> in0 delivery / ring forwarding limiting
#   full improves w/ noreduce          -> split-K reduction / tail imbalance limiting
#   compute-only (skipin0+skipin1) still > ideal DRAM time -> compute feed / block depth limiting
#   all ablations keep a similar floor -> fixed setup/sync overhead limiting
#
# Usage: python3 oracle_ablate.py M K N Ns Pk Sm kb nsb [flag ...]   (one run; prints ABLJSON)
import csv, json, os, statistics, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = f"{ROOT}/build_Release/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
BIN_CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
NUM_TESTS = 6  # run 0 = warmup/compile (dropped); 1.. timed


def parse_csv():
    """per-RISC (zone-tag) -> [median-over-timed-runs cycles for each core]; wall = max over all zones/run."""
    rows = list(csv.reader(open(BIN_CSV)))
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[1], row[2], row[3], row[10].strip())].append((row[11].strip(), int(row[5])))
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
            dur[k] = ds[1:] if len(ds) > 1 else ds  # drop run 0
    by_risc = defaultdict(list)
    for (x, y, z, zone), ds in dur.items():
        by_risc[zone[: -len("-KERNEL")]].append(statistics.median(ds))
    nruns = min((len(v) for v in dur.values()), default=0)
    wall = [max(v[i] for v in dur.values()) for i in range(nruns)] if nruns else []
    percore = {f"{x},{y}": {"zone": zone[:-7], "med_cyc": statistics.median(ds)} for (x, y, z, zone), ds in dur.items()}
    return by_risc, wall, percore


def parse_coremap(text):
    """physical 'x,y' -> {bank,kslice,nslice,mslice,redpos,noc} from RA_COREMAP log lines."""
    cm = {}
    for line in text.splitlines():
        if "COREMAP" not in line:
            continue
        toks = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                if v.lstrip("-").isdigit():
                    toks[k] = int(v)
        if "x" in toks and "y" in toks:
            cm[f"{toks['x']},{toks['y']}"] = {
                k: toks[k] for k in ("bank", "kslice", "nslice", "mslice", "redpos", "noc") if k in toks
            }
    return cm


def run(M, K, N, Ns, Pk, Sm, kb, nsb, flags, coremap=False):
    try:
        os.remove(BIN_CSV)
    except OSError:
        pass
    args = [
        BIN,
        "--unified",
        "--m",
        M,
        "--k",
        K,
        "--n",
        N,
        "--ksplit",
        Pk,
        "--nslice",
        Ns,
        "--msplit",
        Sm,
        "--kb",
        kb,
        "--nsb",
        nsb,
        "--num-tests",
        NUM_TESTS,
    ]
    args = [str(a) for a in args] + list(flags)
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    if coremap:
        env["RA_COREMAP"] = "1"
    r = subprocess.run(args, env=env, capture_output=True, text=True, cwd=ROOT, timeout=300)
    passed = "PASS" in r.stdout or "PASS" in r.stderr
    by_risc, wall, percore = parse_csv()
    wall_us = statistics.median(wall) / FREQ * 1e6 if wall else None
    risc = {
        tag: {
            "n": len(c),
            "max_us": max(c) / FREQ * 1e6,
            "med_us": statistics.median(c) / FREQ * 1e6,
            "min_us": min(c) / FREQ * 1e6,
        }
        for tag, c in sorted(by_risc.items())
    }
    out = {
        "cfg": [Ns, Pk, Sm, kb, nsb],
        "flags": list(flags),
        "pass": passed,
        "wall_us": wall_us,
        "risc": risc,
        "returncode": r.returncode,
        "stderr_tail": (r.stderr[-400:] if not wall else ""),
    }
    if coremap:
        cm = parse_coremap(r.stdout + "\n" + r.stderr)
        for xy, pc in percore.items():
            pc["us"] = pc.pop("med_cyc") / FREQ * 1e6
            if xy in cm:
                pc["role"] = cm[xy]
        out["percore"] = percore
    return out


if __name__ == "__main__":
    # optional leading --coremap toggles RA_COREMAP + role-mapped per-core dump
    argv = sys.argv[1:]
    coremap = False
    if argv and argv[0] == "--coremap":
        coremap = True
        argv = argv[1:]
    M, K, N, Ns, Pk, Sm, kb, nsb = (int(x) for x in argv[:8])
    flags = argv[8:]
    out = run(M, K, N, Ns, Pk, Sm, kb, nsb, flags, coremap=coremap)
    print("ABLJSON " + json.dumps(out))
    w = out["wall_us"]
    fl = " ".join(flags) if flags else "(full)"
    print(
        f"{M}x{K}x{N} cfg={out['cfg']} {fl:32} wall={w:.2f}us pass={out['pass']}"
        if w
        else f"{M}x{K}x{N} {fl}: NO PROFILE (rc={out['returncode']}) {out['stderr_tail'][:200]}"
    )
    for tag, s in out["risc"].items():
        print(f"    {tag:<8} n={s['n']:<4} max={s['max_us']:.2f} med={s['med_us']:.2f} min={s['min_us']:.2f}")
