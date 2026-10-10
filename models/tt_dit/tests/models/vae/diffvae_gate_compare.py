# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare a DiffVAE gate PCC ledger against a committed baseline.

Catches "still green but regressing": a gate can pass its own floor (e.g. 0.999) yet drop
meaningfully below the last recorded run. Fails if any gate's PCC fell more than ``--tol`` below
the baseline, or if a baselined gate is missing from the ledger (it did not run).

Usage:
    diffvae_gate_compare.py <ledger.jsonl> <baseline.json> [--tol 0.0002] [--record]
    --record  : write the ledger's PCCs as the new baseline (do this on a known-good run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_ledger(path: str) -> dict[str, float]:
    """Last-wins PCC per test id from an append-only JSONL ledger."""
    latest: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            latest[rec["test"]] = float(rec["pcc"])
    return latest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("baseline")
    ap.add_argument("--tol", type=float, default=0.0002, help="max allowed PCC drop below baseline")
    ap.add_argument("--record", action="store_true", help="write the ledger as the new baseline and exit")
    args = ap.parse_args()

    current = load_ledger(args.ledger)
    if not current:
        print("no PCC records in ledger (no assert_quality-based gate ran)")
        return 0

    if args.record or not Path(args.baseline).exists():
        Path(args.baseline).write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"recorded baseline: {len(current)} entries -> {args.baseline}")
        return 0

    baseline = json.loads(Path(args.baseline).read_text())
    regressions: list[str] = []
    print(f"{'test':78s} {'baseline':>10s} {'current':>10s} {'delta':>10s}")
    for test in sorted(set(baseline) | set(current)):
        b, c = baseline.get(test), current.get(test)
        if b is None:
            print(f"{test:78s} {'--':>10s} {c * 100:9.4f}%   (new, not baselined)")
        elif c is None:
            print(f"{test:78s} {b * 100:9.4f}% {'MISSING':>10s}   (did not run!)")
            regressions.append(test)
        else:
            d = c - b
            flag = "   REGRESSION" if d < -args.tol else ""
            print(f"{test:78s} {b * 100:9.4f}% {c * 100:9.4f}% {d * 100:+9.4f}%{flag}")
            if d < -args.tol:
                regressions.append(test)

    if regressions:
        print(f"\nFAIL: {len(regressions)} PCC regression(s)/missing beyond tol={args.tol}")
        return 1
    print("\nOK: no PCC regressions vs baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
