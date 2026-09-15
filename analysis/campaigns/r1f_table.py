#!/usr/bin/env python3
"""Render r1f_counters_table.csv (r1f_counters.py) as the markdown table used in bh/campaign_r1.md section 9.
One row per tag: walls of run_idx 1 and 2, wall-core counters of run_idx 1, means over active cores of run_idx 1,
overlap = (FPU + SFPU - MATH) / min(FPU, SFPU) and MATH busy share of the wall (wall core, run_idx 1). 1350 MHz."""
import csv, os
from pathlib import Path

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = os.environ.get("DD") or str(Path.cwd() if _REPO else _SD)
rows = [r for r in csv.reader(open(f"{DD}/r1f_counters_table.csv")) if r and not r[0].startswith("#")]
h = rows[0]
d = [dict(zip(h, r)) for r in rows[1:]]
order = [
    "r1f_causal_q64k128",
    "r1f_causal_q128k128",
    "r1f_causal_q128k256",
    "r1f_causal_q128k512",
    "r1f_causal_q256k128",
    "r1f_causal_q512k128",
    "r1f_causal_q512k512",
    "r1f_noncausal_q64k128",
    "r1f_noncausal_q128k128",
    "r1f_noncausal_q128k256",
    "r1f_noncausal_q128k512",
    "r1f_noncausal_q256k128",
    "r1f_noncausal_q512k128",
    "r1f_noncausal_q512k512",
    "r1f_causal_q128k128_lofi",
    "r1f_causal_q128k128_hifi3",
    "r1f_causal_q128k128_expaccurate",
    "r1f_prod_causal_S4096_q256k256_g110",
    "r1f_a10_causal_S4096_q256k256_g64_fp32off",
    "r1c_causal_S4096_q128k128_hd64",
]


def f0(x):
    return f"{float(x):,.0f}"


out = [
    "| tag | cores | wall run 1 / run 2 (cycles) | wall run 1 (us) | wall core run 1 / run 2 | FPU wall core | SFPU wall core | MATH wall core | FPU mean | SFPU mean | MATH mean | overlap | MATH busy (percent of wall) |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
stats = []
for base in order:
    tag = base + "_zoff_mp"
    r1 = [r for r in d if r["tag"] == tag and r["run_idx"] == "1"][0]
    r2 = [r for r in d if r["tag"] == tag and r["run_idx"] == "2"][0]
    w1, w2 = int(r1["wall_cycles"]), int(r2["wall_cycles"])
    out.append(
        f"| {base} | {r1['active_cores']} | {w1:,} / {w2:,} | {w1/1350:,.1f} | {r1['wall_core']} / {r2['wall_core']} | "
        f"{f0(r1['FPU_COUNTER_wallcore'])} | {f0(r1['SFPU_COUNTER_wallcore'])} | {f0(r1['MATH_COUNTER_wallcore'])} | "
        f"{f0(r1['FPU_COUNTER_mean_active'])} | {f0(r1['SFPU_COUNTER_mean_active'])} | {f0(r1['MATH_COUNTER_mean_active'])} | "
        f"{float(r1['overlap_wallcore']):.3f} | {float(r1['math_pct_wall_wallcore']):.1f} |"
    )
    stats.append(
        (
            base,
            abs(float(r1["FPU_COUNTER_wallcore"]) - float(r2["FPU_COUNTER_wallcore"])),
            abs(float(r1["SFPU_COUNTER_wallcore"]) - float(r2["SFPU_COUNTER_wallcore"])),
            abs(float(r1["MATH_COUNTER_wallcore"]) - float(r2["MATH_COUNTER_wallcore"]))
            / float(r1["MATH_COUNTER_wallcore"])
            * 100,
            abs(w1 - w2) / w1 * 100,
        )
    )
open(f"{DD}/r1f_table.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
print(
    "\nrun 1 vs run 2 on the wall core: max |dFPU| =",
    max(s[1] for s in stats),
    "counts; max |dSFPU| =",
    max(s[2] for s in stats),
    "counts; max |dMATH| = %.2f percent (%s); max wall spread = %.2f percent (%s)"
    % (
        max(s[3] for s in stats),
        max(stats, key=lambda s: s[3])[0],
        max(s[4] for s in stats),
        max(stats, key=lambda s: s[4])[0],
    ),
)
