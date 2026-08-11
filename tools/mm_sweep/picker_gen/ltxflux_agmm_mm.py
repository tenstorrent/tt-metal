#!/usr/bin/env python3
"""Regime-A MM leg of the unfused AG + MM composition, for every LTX/FLUX AGMM shape.

WHAT THIS IS FOR: the intermediate milestone is "unfused AG + optimized regime-A MM" as a replacement for
main's fused AGMM. The AG leg and main's numbers already exist (agmm/comparison.csv on cglagovich/agmm_analysis:
agmm_us = main fused, ag_us = isolated all-gather, mm_us = existing MM, serial_us = ag+mm). The only missing
leg is regime-A MM at the same shapes, which is what this collects.

The MM after an all-gather is FULL-K and data-parallel -- every device runs the identical [M,K]x[K,N] -- so a
single-chip measurement is the right one, and it is what prod_sweep_worker.py does (unit device, fabric off,
config=None so the production picker chooses).

DRAM % uses this campaign's accounting, matching prod_sweep_report.py exactly so numbers stay comparable:
    bytes = Ns*M*K*2 + K*N*2 + M*N*2      vs PEAK = 512 GB/s
`Ns` (n_slices) is the only term that duplicates in0 -- each of the Ns n-slice groups reads all of in0 once --
so it is read from the picker's own log line rather than assumed to be 1.

usage: ltxflux_agmm_mm.py <sweep_shapes.json> [nblocks] [out.jsonl]
"""
import json
import os
import re
import subprocess
import sys

WT = os.environ.get("TT_METAL_HOME", os.getcwd())
SPEC = sys.argv[1]
NBLOCKS = sys.argv[2] if len(sys.argv) > 2 else "3"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ltxflux_mm.jsonl"
PEAK = 512.0  # GB/s, the ceiling used throughout this campaign (prod_sweep_report.py)

CFG_RE = re.compile(r"regime_a_cfg M=(\d+) K=(\d+) N=(\d+) pick=\((\d+),(\d+),(\d+),(\d+),(\d+)\) cores=(\d+)")

shapes = json.load(open(SPEC))
# The MM leg depends only on (M,K,N): device_config picks the AG axis, not the matmul, and fusion is measured
# separately. Dedupe so a shape appearing under both LTX stages / with a fusion variant is run once.
seen, work = set(), []
for s in shapes:
    key = (s["M"], s["K"], s["N"])
    if key in seen:
        continue
    seen.add(key)
    work.append(s)
print(f"{len(shapes)} shapes -> {len(work)} distinct (M,K,N)", flush=True)

out = open(OUT, "w")
for i, s in enumerate(work, 1):
    M, K, N = s["M"], s["K"], s["N"]
    cmd = [
        "docker",
        "exec",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-w",
        WT,
        "-e",
        f"TT_METAL_HOME={WT}",
        "-e",
        f"PYTHONPATH={WT}/ttnn:{WT}/tools:{WT}",
        "-e",
        "ARCH_NAME=blackhole",
        "-e",
        "TT_METAL_DEVICE_PROFILER=1",
        "-e",
        "TT_REGIME_A_LOG_CFG=1",
        "agmm",
        f"{WT}/python_env/bin/python",
        "tools/mm_sweep/picker_gen/prod_sweep_worker.py",
        str(M),
        str(K),
        str(N),
        NBLOCKS,
        "auto",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        blob = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        blob = ""
    # A timed-out worker leaves a live process inside the container holding the device lock, which makes every
    # LATER shape look like it hangs. Always reap.
    subprocess.run(
        ["docker", "exec", "agmm", "bash", "-lc", "pkill -9 -f prod_sweep_worker; exit 0"],
        capture_output=True,
    )

    rec = {"M": M, "K": K, "N": N, "id": s["id"], "tags": s.get("tags", []), "outcome": "no_output"}
    j = blob.find("SWEEP_JSON")
    if j >= 0:
        rec.update(json.loads(blob[blob.find("{", j) :].splitlines()[0]))
    m = CFG_RE.search(blob)
    if m:
        g = [int(x) for x in m.groups()]
        rec["pick"] = g[3:8]  # Pk,Ns,Sm,kb,nsb
        rec["cores"] = g[8]
        Ns = g[4]
        if rec.get("median_us"):
            byt = (Ns * M * K + K * N + M * N) * 2
            rec["dram_pct"] = round(100.0 * (byt / 1e9) / (rec["median_us"] / 1e6) / PEAK, 1)
    out.write(json.dumps(rec) + "\n")
    out.flush()
    bm = rec.get("block_medians")
    print(
        f"[{i:2d}/{len(work)}] {M:5d}x{K:5d}x{N:5d} {rec.get('outcome','?'):8s} "
        f"med={rec.get('median_us','-')} dram%={rec.get('dram_pct','-')} pcc={rec.get('pcc','-')} "
        f"finite={rec.get('finite','-')} blocks={bm} pick={rec.get('pick','-')}",
        flush=True,
    )
out.close()
print(f"\nwrote {OUT}")
