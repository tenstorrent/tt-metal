#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare each shard's official (board-built) perf data with its shadow (CPU-built) pass.

Exits 1 if an expected shard is missing, was not compared, or differs in build, rows or test failures.
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
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
MIN_CYCLES = 30


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


def expected_shards(matrix_json: str) -> set[tuple[str, int]]:
    out = set()
    for row in json.loads(matrix_json):
        labels = (
            " ".join(row["runs_on"])
            if isinstance(row["runs_on"], list)
            else row["runs_on"]
        )
        arch = (
            "wormhole"
            if "N150" in labels.upper()
            else "blackhole" if "P150" in labels.upper() else None
        )
        if arch:
            out.add((arch, int(row["split_group"])))
    return out


def junit_failures(directory: Path) -> int | None:
    files = sorted(directory.glob("*-shadow-run.xml"))
    if not files:
        return None
    total = 0
    for f in files:
        root = ET.parse(f).getroot()
        suites = [root] if root.tag == "testsuite" else root.iter("testsuite")
        total += sum(
            int(x.get("failures", 0)) + int(x.get("errors", 0)) for x in suites
        )
    return total


def why_missing(sdir: Path | None) -> str:
    status = sdir / "status.json" if sdir else None
    if not status or not status.exists():
        return "no shadow data"
    st = json.loads(status.read_text())
    if not st.get("ran"):
        return (
            f"shadow pass skipped, {st.get('minutes_left')} min left before the timeout"
        )
    failed = [k for k, v in st.get("steps", {}).items() if v == "failure"]
    return (
        f"shadow step failed: {', '.join(failed)}"
        if failed
        else "shadow pass incomplete"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", required=True, type=Path)
    ap.add_argument("--shadow", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--expected-matrix", default="", help="the run's test matrix JSON")
    args = ap.parse_args()

    official = shards(args.official, "perf-data", args.suffix)
    shadow = shards(args.shadow, "llk-shadow", args.suffix)
    expected = (
        expected_shards(args.expected_matrix)
        if args.expected_matrix
        else set(official) | set(shadow)
    )
    report, problems, compared = {}, [], 0
    lines = ["## Shadow compile: board-built vs CPU-built ELFs", ""]
    lines += [
        "| Shard | Build identity | Rows (official / shadow / unmatched) | Shadow test failures | Run type | Median ratio | p5–p95 | Rows >2% | Rows >5% |",
        "|---|---|---|---:|---|---:|---|---:|---:|",
    ]
    for arch, group in sorted(expected | set(official) | set(shadow)):
        name = f"{arch} {group}"
        entry = report.setdefault(name, {})
        sdir, odir = shadow.get((arch, group)), official.get((arch, group))
        if odir is None:
            problems.append(f"{name}: no official data")
            lines.append(f"| {name} | — | no official data | | | | | | |")
            continue
        ident_file = sdir / "identity.json" if sdir else None
        ident = (
            json.loads(ident_file.read_text())
            if ident_file and ident_file.exists()
            else None
        )
        entry["identity"] = ident
        if ident is None:
            reason = why_missing(sdir)
            problems.append(f"{name}: {reason}")
            lines.append(f"| {name} | {reason} | | | | | | | |")
            continue
        bad = ident["different"] + ident["missing_in_tree"] + ident["extra_in_tree"]
        ident_txt = (
            f"{ident['identical']:,} / {ident['compared']:,} identical"
            if not bad
            else f"**{ident['different']} different, {ident['missing_in_tree']} missing, {ident['extra_in_tree']} extra**"
        )
        if bad:
            problems.append(f"{name}: builds differ")
        fails = junit_failures(sdir)
        entry["shadow_test_failures"] = fails
        fails_txt = "no report" if fails is None else str(fails)
        if fails:
            problems.append(f"{name}: {fails} shadow tests failed")
        a, b = load(odir), load(sdir)
        if a is None or b is None:
            reason = "no official parquet" if a is None else why_missing(sdir)
            problems.append(f"{name}: {reason}")
            lines.append(f"| {name} | {ident_txt} | {reason} | {fails_txt} | | | | | |")
            continue
        compared += 1
        cmp_ = compare(a, b)
        entry["cycles"] = cmp_
        unmatched = cmp_["rows_only_official"] + cmp_["rows_only_shadow"]
        if unmatched:
            problems.append(f"{name}: {unmatched} rows in one pass only")
        rows_txt = f"{cmp_['rows_official']:,} / {cmp_['rows_shadow']:,} / {unmatched}"
        for i, (rt, s) in enumerate(cmp_["run_types"].items()):
            first = i == 0
            lines.append(
                f"| {name if first else ''} | {ident_txt if first else ''} | {rows_txt if first else ''} "
                f"| {fails_txt if first else ''} | {rt} | {s['median']:.4f} | {s['p5']:.4f}–{s['p95']:.4f} "
                f"| {s['over_2pct']:.2%} | {s['over_5pct']:.2%} |"
            )
    if compared == 0:
        problems.append("no shard was compared")

    lines += [
        "",
        f"**{'PASS' if not problems else 'CHECK'}**: "
        + (
            "; ".join(problems)
            or f"all {compared} shards have identical builds and the same rows."
        ),
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"problems": problems, "compared": compared, "shards": report}, indent=2
        )
        + "\n"
    )
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
