# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gate the SFPU accuracy sweep against the committed per-arch baseline.

Run from tests/python_tests after the accuracy sweep has written its per-op
Parquet files:

    python -m accuracy.check_accuracy_baseline --arch wormhole
    python -m accuracy.check_accuracy_baseline --arch wormhole --report accuracy_report.md
    python -m accuracy.check_accuracy_baseline --arch wormhole --update-baseline

Exit codes: 0 pass (improvements are listed but pass), 1 regression,
2 schema mismatch (variant set / sweep density changed, or no baseline yet).

--update-baseline rewrites accuracy/baselines/<arch>.csv from the current run
and exits 0. Commit the result; the PR diff is the review of the new numbers.
--from-parquet points at another run's output, e.g. a downloaded CI artifact,
which is how a Blackhole baseline gets refreshed without a Blackhole card.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from accuracy.baseline import (
    ARCH_ABBR,
    compare,
    load_baseline,
    render_report,
    save_baseline,
    summarize_dir,
)

_THIS_DIR = Path(__file__).resolve().parent


def default_parquet_dir(arch: str) -> Path:
    """Where the sweep writes merged per-op Parquet for *arch* (matches accuracy_harness)."""
    return _THIS_DIR / "_csv_output" / ARCH_ABBR[arch]


def default_baseline_path(arch: str) -> Path:
    """The committed reference for *arch*."""
    return _THIS_DIR / "baselines" / f"{ARCH_ABBR[arch]}.csv"


def _parse(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m accuracy.check_accuracy_baseline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", required=True, choices=sorted(ARCH_ABBR))
    p.add_argument(
        "--from-parquet",
        type=Path,
        metavar="DIR",
        help="per-op Parquet directory to summarize (default: accuracy/_csv_output/<arch>)",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        metavar="FILE",
        help="baseline CSV to compare against / rewrite (default: accuracy/baselines/<arch>.csv)",
    )
    p.add_argument(
        "--report",
        type=Path,
        metavar="FILE",
        help="also write the markdown report here (always printed to stdout)",
    )
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline from the current run instead of comparing",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse(argv)
    parquet_dir = args.from_parquet or default_parquet_dir(args.arch)
    baseline_path = args.baseline or default_baseline_path(args.arch)
    abbr = ARCH_ABBR[args.arch]

    try:
        current = summarize_dir(parquet_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.update_baseline:
        save_baseline(current, baseline_path)
        print(f"wrote {len(current)} variant(s) to {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(
            f"error: no baseline at {baseline_path}. Generate one with "
            "--update-baseline and commit it.",
            file=sys.stderr,
        )
        return 2

    result = compare(load_baseline(baseline_path), current)
    report = render_report(result, arch=abbr)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
