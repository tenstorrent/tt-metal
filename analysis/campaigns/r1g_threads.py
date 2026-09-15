#!/usr/bin/env python3
"""R1g: wall-core thread split of the zones-on twins, head_dim 64 (r1g_noncausal_S2048_q128k128_hd64_zon) against head_dim 128
(r1a_noncausal_S2048_q128k128_zon, same shape otherwise). Per RISC and zone: accumulated cycles on the wall-setting core corrected
for the inside-window zone tax (ACC_IN, T0.2: 0 on TRISC_0, 2 cycles per occurrence elsewhere), mean of invocations 1 and 2
(invocation 0 discarded), with the occurrence count and the share of the zones-on device wall. Writes r1g_thread_split.csv and .md."""
import pandas as pd, os, re
from pathlib import Path

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = os.environ.get("DD") or str(Path.cwd() if _REPO else _SD)
ACC_IN = {"TRISC_0": 0.0, "TRISC_1": 2.0, "TRISC_2": 2.0, "BRISC": 2.0, "NCRISC": 2.0}
ORDER = ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
TAGS = {"hd128": "r1a_noncausal_S2048_q128k128", "hd64": "r1g_noncausal_S2048_q128k128_hd64"}


def split(tag):
    runs = pd.read_csv(f"{DD}/{tag}_zon_runs.csv")
    cores = pd.read_csv(f"{DD}/{tag}_zon_cores.csv")
    acc = {}
    its = [it for it in runs.run_idx if it > 0]
    walls, wcs = [], []
    for it in its:
        r = runs[runs.run_idx == it].iloc[0]
        walls.append(r.wall_dev_cycles)
        wcs.append(f"({int(r.wall_core_x)},{int(r.wall_core_y)})")
        wc = cores[(cores.run_idx == it) & (cores.is_wall_core == 1)]
        for _, cr in wc.iterrows():
            acc.setdefault((cr.risc, "KERNEL_DUR"), []).append((cr.kernel_dur, 1))
            for c in wc.columns:
                if c.endswith("_N") and not pd.isna(cr[c]):
                    acc.setdefault((cr.risc, c[:-2]), []).append((cr[c[:-2]] - cr[c] * ACC_IN[cr.risc], cr[c]))
    m = {k: (sum(x for x, _ in v) / len(v), sum(n for _, n in v) / len(v)) for k, v in acc.items()}
    prov = open(f"{DD}/{tag}_zon.csv").readline().strip()
    return m, sum(walls) / len(walls), wcs, prov


res = {k: split(t) for k, t in TAGS.items()}
keys = sorted(set(res["hd128"][0]) | set(res["hd64"][0]), key=lambda k: (ORDER.index(k[0]), k[1] != "KERNEL_DUR", k[1]))
rows = []
for risc, part in keys:
    a = res["hd128"][0].get((risc, part), (float("nan"), 0))
    b = res["hd64"][0].get((risc, part), (float("nan"), 0))
    if (a[0] == 0 or a[0] != a[0]) and (b[0] == 0 or b[0] != b[0]):
        continue
    rows.append(
        dict(
            risc=risc,
            part=part,
            hd128_cycles=round(a[0]),
            hd128_count=round(a[1]),
            hd128_pct_wall=round(100 * a[0] / res["hd128"][1], 2),
            hd64_cycles=round(b[0]),
            hd64_count=round(b[1]),
            hd64_pct_wall=round(100 * b[0] / res["hd64"][1], 2),
            ratio_hd64_hd128=round(b[0] / a[0], 3) if a[0] else float("nan"),
        )
    )
df = pd.DataFrame(rows)
prov = (
    "# PROVENANCE: wall-setting core of the zones-on twins, tax-corrected (ACC_IN) zone sums, mean of invocations 1 and 2; "
    f"hd128 = {TAGS['hd128']}_zon (wall {res['hd128'][1]:.0f} cycles, wall cores {'/'.join(res['hd128'][2])}); "
    f"hd64 = {TAGS['hd64']}_zon (wall {res['hd64'][1]:.0f} cycles, wall cores {'/'.join(res['hd64'][2])}); "
    f"hd128 raw PROVENANCE: {res['hd128'][3][:200]}; hd64 raw PROVENANCE: {res['hd64'][3][:200]}; r1g_threads.py 2026-09-12\n"
)
with open(f"{DD}/r1g_thread_split.csv", "w") as f:
    f.write(prov)
    df.to_csv(f, index=False)
md = [
    "| RISC | zone | hd128 cycles | hd128 count | hd128 pct of wall | hd64 cycles | hd64 count | hd64 pct of wall | hd64 / hd128 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    md.append(
        f"| {r['risc']} | {r['part']} | {r['hd128_cycles']:,} | {r['hd128_count']:,} | {r['hd128_pct_wall']:.1f} | {r['hd64_cycles']:,} | {r['hd64_count']:,} | {r['hd64_pct_wall']:.1f} | {r['ratio_hd64_hd128']:.3f} |"
    )
open(f"{DD}/r1g_thread_split.md", "w").write("\n".join(md) + "\n")
print(prov)
print("\n".join(md))
