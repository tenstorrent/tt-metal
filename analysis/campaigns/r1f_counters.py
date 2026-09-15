#!/usr/bin/env python3
"""R1f counter table: per multipass run and invocation (first discarded), FPU_COUNTER, SFPU_COUNTER, MATH_COUNTER on the
wall-setting core and as the mean over active cores (FPU_COUNTER > 0), the zones-off device wall of the same run, and the
derived shares and overlap. Writes data/bh_zones/r1f_counters_table.csv with a PROVENANCE first line."""
import glob
import os
import re
from pathlib import Path

import pandas as pd

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = os.environ.get("DD") or str(Path.cwd() if _REPO else _SD)
TAGS = sorted(glob.glob(f"{DD}/r1f_*_counters.csv")) + [f"{DD}/r1c_causal_S4096_q128k128_hd64_zoff_mp_counters.csv"]
rows = []
for f in TAGS:
    if not os.path.exists(f):
        continue
    tag = os.path.basename(f).replace("_counters.csv", "")
    c = pd.read_csv(f)
    runs = pd.read_csv(f"{DD}/{tag}_runs.csv")
    prov = open(f"{DD}/{tag}.csv").readline().strip()
    env = re.search(r"env (.*?) mode=", prov).group(1)
    fw = re.search(r"fw_bundle=(\S+);", prov).group(1)
    sha = re.search(r"tt-metal (\w+)", prov).group(1)[:11]
    for _, run in runs.iterrows():
        if run.run_idx == 0:
            continue
        rid = run.run_id
        cc = c[c.run_id == rid]
        fpu = cc[cc.counter == "FPU_COUNTER"]
        active = set(zip(fpu[fpu.value > 0].core_x, fpu[fpu.value > 0].core_y))
        wc = (int(run.wall_core_x), int(run.wall_core_y))
        rec = dict(
            tag=tag,
            config=env,
            run_idx=int(run.run_idx),
            wall_cycles=int(run.wall_dev_cycles),
            wall_core=f"({wc[0]},{wc[1]})",
            active_cores=len(active),
            fw=fw,
            sha=sha,
        )
        for name in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
            sub = cc[cc.counter == name]
            act = sub[[(x, y) in active for x, y in zip(sub.core_x, sub.core_y)]]
            wv = sub[(sub.core_x == wc[0]) & (sub.core_y == wc[1])].value
            rec[f"{name}_wallcore"] = float(wv.iloc[0]) if len(wv) else float("nan")
            rec[f"{name}_mean_active"] = float(act.value.mean()) if len(act) else float("nan")
            rec[f"{name}_max"] = float(act.value.max()) if len(act) else float("nan")
        rec["ref_cnt_mean"] = float(cc[cc.counter == "MATH_COUNTER"].ref_cnt.mean())
        for basis in ("wallcore", "mean_active"):
            f_, s_, m_ = rec[f"FPU_COUNTER_{basis}"], rec[f"SFPU_COUNTER_{basis}"], rec[f"MATH_COUNTER_{basis}"]
            rec[f"overlap_{basis}"] = (f_ + s_ - m_) / min(f_, s_) if min(f_, s_) > 0 else float("nan")
            rec[f"math_pct_wall_{basis}"] = 100 * m_ / rec["wall_cycles"]
        rows.append(rec)
df = pd.DataFrame(rows)
prov = (
    "# PROVENANCE: p100a fw 19.9.0, 1350 MHz; tt-metal calibration checkout scratch/sdpa_zones f4ab8088c8f, kernels = 72620d5 with the zone "
    "patch compiled out (SDPA_ZONES 0; the Tensix instruction stream of the compiled-out build equals the unmodified 72620d5 build, T2.1 "
    "section 4.6 cross-check); python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest analysis/zone_sweep.py; "
    "3 invocations per process, run_idx 0 discarded; counters per core from the 9090 records (post_process.load_counters), wall = device wall "
    "of the same multipass run (KERNEL zones); overlap = (FPU + SFPU - MATH) / min(FPU, SFPU); 2026-09-12\n"
)
with open(f"{DD}/r1f_counters_table.csv", "w") as fh:
    fh.write(prov)
    df.to_csv(fh, index=False)
pd.set_option("display.width", 300)
cols = [
    "tag",
    "run_idx",
    "wall_cycles",
    "wall_core",
    "active_cores",
    "FPU_COUNTER_wallcore",
    "SFPU_COUNTER_wallcore",
    "MATH_COUNTER_wallcore",
    "FPU_COUNTER_mean_active",
    "SFPU_COUNTER_mean_active",
    "MATH_COUNTER_mean_active",
    "overlap_wallcore",
    "overlap_mean_active",
    "math_pct_wall_wallcore",
]
print(df[cols].round(3).to_string(index=False))
