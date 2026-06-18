#!/usr/bin/env python3
"""Triage a hardware-determinism sweep.

Reads the JSONL report produced by the determinism harness in conftest.py
(see TT_LLK_DETERMINISM_RUNS) and reports which variants were nondeterministic
run-to-run, so they can be re-run at a higher iteration count for confirmation.

Usage:
    python determinism_triage.py [report.jsonl] [--out flaky_nodeids.txt]

A variant is "flaky" when it produced more than one distinct result hash
across the N runs of a single sweep. The --out file lists one pytest node id
per line (feed each back through run_test.sh --test-id for the deep-dive).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Variants whose non-determinism is BY DESIGN and must not be flagged as a
# hardware bug. Stochastic rounding draws a PRNG to choose rounding direction,
# so its output legitimately differs run-to-run. Matched against the pytest
# node id. Validated on test_unpack_A.py: every nondeterministic variant there
# was stoch_rnd in {Pack, All}; stoch_rnd in {No, Fpu} were all deterministic.
# Extend this list if other intentional-nondeterminism knobs surface (dropout,
# random init, etc.).
EXPECTED_NONDET_PATTERNS = [
    re.compile(r"stoch_rnd_(?!No)\w+"),  # stoch_rnd_Pack / _All / _Fpu (anything but _No)
]


def is_expected_nondet(record: dict) -> bool:
    # Stochastic rounding is intentionally non-deterministic and is NOT
    # reproducible from software on WH B0 (the PRNG seeder cannot be
    # re-triggered by a Tensix-core CFG write), so it is always excused.
    return any(p.search(record.get("nodeid") or "") for p in EXPECTED_NONDET_PATTERNS)


def load_records(path: Path):
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: {path}:{line_no}: skipping bad line: {e}", file=sys.stderr)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", default="determinism_report.jsonl",
                    help="path to the JSONL report (default: determinism_report.jsonl)")
    ap.add_argument("--out", default="flaky_nodeids.txt",
                    help="write suspicious (non-intentional) flaky node ids here (default: flaky_nodeids.txt)")
    ap.add_argument("--include-expected", action="store_true",
                    help="also treat intentional non-determinism (stochastic rounding) as flaky")
    args = ap.parse_args()

    report = Path(args.report)
    if not report.exists():
        print(f"error: report not found: {report}", file=sys.stderr)
        return 2

    records = load_records(report)
    if not records:
        print("error: report is empty", file=sys.stderr)
        return 2

    # De-dup by (nodeid, variant_id): keep the record with the most runs
    # (a later, higher-N sweep supersedes an earlier triage pass).
    best = {}
    for r in records:
        key = (r.get("nodeid"), r.get("variant_id"))
        prev = best.get(key)
        if prev is None or r.get("runs", 0) >= prev.get("runs", 0):
            best[key] = r
    deduped = list(best.values())

    nondet = [r for r in deduped if not r.get("deterministic", True)]
    expected = [r for r in nondet if is_expected_nondet(r)]
    suspicious = [r for r in nondet if not is_expected_nondet(r)]
    # The set we report as flaky depends on --include-expected.
    flaky = nondet if args.include_expected else suspicious
    flaky.sort(key=lambda r: r.get("n_distinct", 0), reverse=True)

    by_file = defaultdict(lambda: [0, 0])  # test_name -> [total, suspicious]
    for r in deduped:
        name = r.get("test_name") or "(unknown)"
        by_file[name][0] += 1
        if r in suspicious:
            by_file[name][1] += 1

    total = len(deduped)
    print(f"variants measured           : {total}")
    print(f"deterministic               : {total - len(nondet)}")
    print(f"nondeterministic (total)    : {len(nondet)}")
    print(f"  - expected (stoch. round) : {len(expected)}")
    print(f"  - SUSPICIOUS (investigate): {len(suspicious)}")
    print()

    if suspicious:
        print("=== SUSPICIOUS nondeterministic variants (by distinct-hash count) ===")
        for r in sorted(suspicious, key=lambda r: r.get("n_distinct", 0), reverse=True):
            print(f"  [{r.get('n_distinct')}/{r.get('runs')} distinct] {r.get('nodeid')}")
        print()
        print("=== test files with suspicious non-determinism ===")
        for name, (tot, fl) in sorted(by_file.items(), key=lambda kv: kv[1][1], reverse=True):
            if fl:
                print(f"  {fl:>4}/{tot:<4} suspicious  {name}")
        print()
    else:
        print("No suspicious non-determinism — all non-determinism is intentional (stochastic rounding).")
        print()

    nodeids = sorted({r.get("nodeid") for r in flaky if r.get("nodeid")})
    out = Path(args.out)
    out.write_text("\n".join(nodeids) + ("\n" if nodeids else ""))
    print(f"wrote {len(nodeids)} flaky node id(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
