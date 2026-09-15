#!/usr/bin/env python3
"""Append card_log rows for every r1*.csv raw file not yet logged (idempotent by tag)."""
import glob, os, re
from pathlib import Path

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = os.environ.get("DD") or str(Path.cwd() if _REPO else _SD)
HANDOFF = os.environ.get("HANDOFF", DD if _REPO else os.path.dirname(os.path.dirname(DD)))
LOG = os.path.join(HANDOFF, "bh", "card_log.md")
done = set(re.findall(r"data/bh_zones/(r1\S+?)\.csv", open(LOG).read()))
rows = []
for f in sorted(glob.glob(f"{DD}/r1*.csv"), key=os.path.getmtime):
    tag = os.path.basename(f)[:-4]
    if (
        any(tag.endswith(s) for s in ("_runs", "_cores", "_raw", "_counters", "_ops_perf_results", "_table", "_walls"))
        or tag in done
    ):
        continue
    prov = open(f).readline().strip()
    t0 = re.search(r"; (2026-\S+); cmd", prov)
    cmd = re.search(r"cmd: (.*?); env", prov)
    env = re.search(r"env (.*?)(?: mode=|; zone_config|$)", prov)
    fw = re.search(r"fw_bundle=(\S+);", prov)
    sha = re.search(r"(tt-metal(?:-fresh)? \w+)", prov)
    zc = "zones ON" if "SDPA_ZONES 1" in prov else "zones off"
    mode = "mp" if "mode=mp" in prov else "plain"
    rf = f"{DD}/{tag}_runs.csv"
    walls = ""
    if os.path.exists(rf):
        import pandas as pd

        r = pd.read_csv(rf)
        walls = (
            ", ".join(str(int(w)) for w in r.wall_dev_cycles)
            + " (wall cores "
            + "/".join(f"({x},{y})" for x, y in zip(r.wall_core_x, r.wall_core_y))
            + ")"
        )
    block = tag.split("_")[0].upper()
    rows.append(
        f"| 21 | {t0.group(1) if t0 else ''} | R1 {block} | `{cmd.group(1) if cmd else ''}` | {env.group(1)[:220] if env else ''} | {zc}; {sha.group(1) if sha else ''}; fw {fw.group(1) if fw else ''}; {mode} | data/bh_zones/{tag}.csv | {walls} |"
    )
if rows:
    open(LOG, "a").write("\n".join(rows) + "\n")
print(len(rows), "rows appended")
