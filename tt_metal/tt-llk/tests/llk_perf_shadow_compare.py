#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare a nightly's board-built measurements with its CPU-built shadow pass.

In shadow mode each board shard measures twice: first the ELFs it compiled
itself (the official data), then the ELFs a CPU runner compiled for the same
shard. This script reads both, shard by shard:

  - build identity: the board's own build against the CPU build's checksums
    (written on the board by llk_perf_offload.py verify-tree);
  - measured rows: the same set of rows in both passes;
  - cycles: the ratio shadow / official for every row and run type.

It writes a Markdown report (to $GITHUB_STEP_SUMMARY when set) and a JSON file.
It exits 1 when a shard has no shadow data, the builds differ, or the row sets
differ. Cycle ratios are reported, not judged: two passes on one board differ
by measurement noise only.

Usage: llk_perf_shadow_compare.py --official DIR --shadow DIR --out FILE.json
  DIR/official: perf-data-<arch>-<group>/**/*.parquet   (the nightly's artifacts)
  DIR/shadow:   llk-shadow-<arch>-<group>/{perf/**/*.parquet,identity.json}
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = {
    "test_name",
    "commit_sha",
    "arch",
    "run_id",
    "timestamp",
    "pipeline",
    "pr_number",
}
MIN_CYCLES = 30  # Ratios below this are dominated by a cycle or two of jitter.


def shards(root: Path, prefix: str, suffix: str) -> dict[tuple[str, int], Path]:
    found = {}
    for d in root.iterdir() if root.exists() else []:
        m = re.fullmatch(rf"{prefix}-([a-z]+)-(\d+){re.escape(suffix)}", d.name)
        if m and d.is_dir():
            found[(m.group(1), int(m.group(2)))] = d
    return found


def load(directory: Path) -> pd.DataFrame | None:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    metrics = [c for c in df.columns if c.startswith(("mean(", "std("))]
    keys = [c for c in df.columns if c not in metrics and c not in META_COLUMNS]
    df["_key"] = (
        df["test_name"].astype(str) + "|" + df[keys].astype(str).agg("|".join, axis=1)
    )
    return df.drop_duplicates("_key").set_index("_key")


def compare(official: pd.DataFrame, shadow: pd.DataFrame) -> dict:
    common = official.index.intersection(shadow.index)
    out = {
        "rows_official": len(official),
        "rows_shadow": len(shadow),
        "rows_only_official": len(official.index.difference(shadow.index)),
        "rows_only_shadow": len(shadow.index.difference(official.index)),
        "run_types": {},
    }
    for col in [c for c in official.columns if c.startswith("mean(")]:
        if col not in shadow.columns:
            continue
        a, b = official.loc[common, col], shadow.loc[common, col]
        ok = a.notna() & b.notna() & (a >= MIN_CYCLES)
        if not ok.any():
            continue
        r = (b[ok] / a[ok]).to_numpy()
        dev = np.abs(r - 1)
        out["run_types"][col[5:-1]] = {
            "n": int(ok.sum()),
            "median": float(np.median(r)),
            "p5": float(np.percentile(r, 5)),
            "p95": float(np.percentile(r, 95)),
            "exact": float(np.mean(r == 1)),
            "over_2pct": float(np.mean(dev > 0.02)),
            "over_5pct": float(np.mean(dev > 0.05)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", required=True, type=Path)
    ap.add_argument("--shadow", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--suffix", default="", help="the run's artifact suffix")
    args = ap.parse_args()

    official = shards(args.official, "perf-data", args.suffix)
    shadow = shards(args.shadow, "llk-shadow", args.suffix)
    report, problems = {}, []
    lines = ["## Shadow compile: board-built vs CPU-built ELFs", ""]
    lines += [
        "| Shard | Build identity | Rows (official / shadow / unmatched) | Run type | Median ratio | p5–p95 | Rows >2% | Rows >5% |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for arch, group in sorted(official):
        name = f"{arch} {group}"
        entry = report.setdefault(name, {})
        sdir = shadow.get((arch, group))
        if sdir is None:
            problems.append(f"{name}: no shadow data")
            lines.append(f"| {name} | — | no shadow data | | | | | |")
            continue
        ident_file = sdir / "identity.json"
        ident = json.loads(ident_file.read_text()) if ident_file.exists() else None
        entry["identity"] = ident
        if ident is None:
            ident_txt = "not checked"
            problems.append(f"{name}: no build identity result")
        else:
            bad = ident["different"] + ident["missing_in_tree"] + ident["extra_in_tree"]
            ident_txt = (
                f"{ident['identical']:,} / {ident['compared']:,} identical"
                if not bad
                else f"**{ident['different']} different, {ident['missing_in_tree']} missing, {ident['extra_in_tree']} extra**"
            )
            if bad:
                problems.append(f"{name}: builds differ")
        a, b = load(official[(arch, group)]), load(sdir)
        if a is None or b is None:
            problems.append(
                f"{name}: no parquet in {'official' if a is None else 'shadow'} data"
            )
            lines.append(f"| {name} | {ident_txt} | missing parquet | | | | | |")
            continue
        cmp_ = compare(a, b)
        entry["cycles"] = cmp_
        unmatched = cmp_["rows_only_official"] + cmp_["rows_only_shadow"]
        if unmatched:
            problems.append(f"{name}: {unmatched} rows in one pass only")
        rows_txt = f"{cmp_['rows_official']:,} / {cmp_['rows_shadow']:,} / {unmatched}"
        for i, (rt, s) in enumerate(cmp_["run_types"].items()):
            lines.append(
                f"| {name if i == 0 else ''} | {ident_txt if i == 0 else ''} | {rows_txt if i == 0 else ''} "
                f"| {rt} | {s['median']:.4f} | {s['p5']:.4f}–{s['p95']:.4f} | {s['over_2pct']:.2%} | {s['over_5pct']:.2%} |"
            )
    for arch, group in sorted(set(shadow) - set(official)):
        problems.append(f"{arch} {group}: shadow data without official data")

    lines += [
        "",
        f"**{'PASS' if not problems else 'CHECK'}**: "
        + (
            "; ".join(problems) or "every shard has identical builds and the same rows."
        ),
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"problems": problems, "shards": report}, indent=2) + "\n"
    )
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
