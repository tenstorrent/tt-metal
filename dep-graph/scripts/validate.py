"""Validation harness: run expected_chains.yaml against dep-graph.sqlite.

Each chain runs a single SQL query that must return an integer (typically a
COUNT). The harness checks `expect_min` / `expect_max` / `expect_exact`
constraints. Exit code 0 on all-pass, 1 on any fail.

Usage:
    python dep-graph/scripts/validate.py
    python dep-graph/scripts/validate.py --db <path> --chains <path>
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("error: PyYAML not installed. Install with `uv pip install pyyaml`.", file=sys.stderr)
    sys.exit(2)


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def run_chain(con: sqlite3.Connection, chain: dict) -> tuple[bool, str]:
    """Run one chain; return (passed, message)."""
    name = chain.get("name", "<unnamed>")
    query = chain.get("query")
    if not query:
        return False, f"chain {name!r} missing `query`"
    try:
        row = con.execute(query).fetchone()
    except sqlite3.Error as e:
        return False, f"SQL error: {e}"
    if row is None or len(row) != 1:
        return False, "query must return exactly one column"
    value = row[0]
    if not isinstance(value, (int, float)):
        return False, f"query returned non-numeric: {value!r}"
    actual = int(value)

    failures = []
    if "expect_min" in chain and actual < chain["expect_min"]:
        failures.append(f"got {actual}, expected >= {chain['expect_min']}")
    if "expect_max" in chain and actual > chain["expect_max"]:
        failures.append(f"got {actual}, expected <= {chain['expect_max']}")
    if "expect_exact" in chain and actual != chain["expect_exact"]:
        failures.append(f"got {actual}, expected == {chain['expect_exact']}")

    if failures:
        return False, "; ".join(failures) + f"  (actual={actual})"
    return True, f"ok  (actual={actual})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/workspace/dep-graph/out/dep-graph.sqlite")
    ap.add_argument("--chains", default=str(Path(__file__).resolve().parent.parent / "tests" / "expected_chains.yaml"))
    ap.add_argument("-q", "--quiet", action="store_true", help="Print only failures and the summary")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.chains).exists():
        print(f"chains file not found: {args.chains}", file=sys.stderr)
        sys.exit(2)

    doc = yaml.safe_load(Path(args.chains).read_text())
    chains = doc.get("chains", []) if isinstance(doc, dict) else []
    if not chains:
        print("error: no chains found", file=sys.stderr)
        sys.exit(2)

    con = sqlite3.connect(args.db)
    n_pass = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []

    for ch in chains:
        passed, msg = run_chain(con, ch)
        if passed:
            n_pass += 1
            if not args.quiet:
                print(f"{GREEN}PASS{RESET}  {ch['name']}  {DIM}{msg}{RESET}")
        else:
            n_fail += 1
            failures.append((ch["name"], msg))
            print(f"{RED}FAIL{RESET}  {ch['name']}  {RED}{msg}{RESET}")
            if ch.get("description"):
                print(f"      {DIM}{ch['description'].strip()}{RESET}")

    total = n_pass + n_fail
    color = GREEN if n_fail == 0 else RED
    print()
    print(f"{color}{n_pass}/{total} chains passed{RESET}")
    if n_fail:
        print(f"{RED}failures:{RESET}")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
