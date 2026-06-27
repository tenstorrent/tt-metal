#!/usr/bin/env python3
"""Rank frontier Table 2 misses and print safe replay/disassembly commands.

The report is intentionally read-only. It does not run silicon and it does not
write result artifacts. Use the emitted commands one at a time, or with distinct
TT_VISIBLE_DEVICES plus distinct tt-metal checkouts if sharding manually.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results" / "frontier" / "table2" / "table2_frontier_ttnn.csv"
DEFAULT_OUTDIR = ROOT / "results" / "frontier" / "analysis"


def f(value: str | None) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def load_rows(path: Path, dtype: str) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("dtype") == dtype]
    return rows


def classify(row: dict[str, str]) -> str:
    result = row.get("audited_result", "")
    if result:
        return result
    ours_ulp, theirs_ulp = f(row.get("max_ulp")), f(row.get("ttnn_maxulp"))
    ours_us, theirs_us = f(row.get("runtime_us")), f(row.get("ttnn_us"))
    if None in (ours_ulp, theirs_ulp, ours_us, theirs_us):
        return "incomplete"
    if ours_ulp <= theirs_ulp and ours_us <= theirs_us:
        return "win_both"
    if ours_ulp <= theirs_ulp:
        return "accuracy_match_slow"
    if ours_us < theirs_us:
        return "faster_less_accurate"
    return "loss"


def miss_score(row: dict[str, str]) -> tuple[float, float, str]:
    ours_us, theirs_us = f(row.get("runtime_us")), f(row.get("ttnn_us"))
    ours_ulp, theirs_ulp = f(row.get("max_ulp")), f(row.get("ttnn_maxulp"))
    runtime_gap = (ours_us - theirs_us) if ours_us is not None and theirs_us is not None else -1.0
    ulp_gap = (ours_ulp - theirs_ulp) if ours_ulp is not None and theirs_ulp is not None else -1.0
    return (runtime_gap, ulp_gap, row.get("activation", ""))


def replay_command(row: dict[str, str], dtype: str, outdir: Path, cache: Path, chip: str) -> str:
    act = row["activation"]
    coeff = row["coeff_csv"]
    stem = Path(row["csv"]).stem
    dis = outdir / "disassembly" / dtype / f"{act}_{stem}.dis"
    run_log = outdir / "logs" / dtype / f"{act}_{stem}.log"
    return " ".join(
        [
            f"TT_VISIBLE_DEVICES={q(chip)}",
            f"TT_METAL_CACHE={q(cache)}",
            "TT_METAL_FORCE_JIT_COMPILE=1",
            "TT_METAL_KERNEL_MAP=1",
            "TT_METAL_BACKEND_DUMP_RUN_CMD=1",
            f"TT_POLY_FIT_DIR={q(Path.home() / 'tt-polynomial-fitter')}",
            q(ROOT / "run_csv.sh"),
            q(coeff),
            "--activation",
            q(act),
            "--precision",
            q(dtype),
            "--tiles",
            "256",
            "--runs",
            "3",
            f"2>&1 | tee {q(run_log)}",
            "&&",
            q(ROOT / "tools" / "disassemble_adhoc.sh"),
            "--cache",
            q(cache),
            "--out",
            q(dis),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/tt-metal-cache-frontier-inspect"))
    parser.add_argument("--chip", default="0")
    parser.add_argument("--include-waived", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.table, args.dtype)
    misses = []
    for row in rows:
        cls = classify(row)
        row["_class"] = cls
        if cls in {"accuracy_match_slow", "faster_less_accurate", "loss"}:
            misses.append(row)
        elif args.include_waived and cls == "waived_fast":
            misses.append(row)

    misses.sort(key=miss_score, reverse=True)

    print(f"table: {args.table}")
    print(f"dtype: {args.dtype}")
    print(f"misses: {len(misses)}")
    print()
    for row in misses:
        ours_us, theirs_us = f(row.get("runtime_us")), f(row.get("ttnn_us"))
        ours_ulp, theirs_ulp = f(row.get("max_ulp")), f(row.get("ttnn_maxulp"))
        gap = (ours_us - theirs_us) if ours_us is not None and theirs_us is not None else None
        speedup = f(row.get("speedup_vs_ttnn"))
        print(
            "{activation:14s} {klass:22s} {method:24s} cfg=s{segments}/d{degree} "
            "ulp={ulp}/{tulp} us={us}/{tus} gap={gap} speedup={speedup} csv={csv}".format(
                activation=row.get("activation", ""),
                klass=row["_class"],
                method=row.get("method", ""),
                segments=row.get("segments", ""),
                degree=row.get("degree", ""),
                ulp=ours_ulp,
                tulp=theirs_ulp,
                us=ours_us,
                tus=theirs_us,
                gap="--" if gap is None else f"{gap:.3f}",
                speedup="--" if speedup is None else f"{speedup:.3f}",
                csv=row.get("csv", ""),
            )
        )
        print("  " + replay_command(row, args.dtype, args.outdir, args.cache, args.chip))


if __name__ == "__main__":
    main()
