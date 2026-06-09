"""Query allocated test time for a (team, test type, machine) and compare it to
the budget declared in .github/time_budget.yaml.

The companion script verify_time_budget.py checks one tests file against the
budget during CI. This tool answers the inverse question: "How much time does
team X currently allocate for test type Y on machine Z?" by summing the
per-SKU timeouts across every tests file of that test type.

Notes on the data model (see tests/pipeline_reorg/*_tests.yaml):
  * Each entry has a `team:` field. This field -- not the file name -- is the
    authoritative owner. A single file (e.g. galaxy_unit_tests.yaml) mixes
    entries from several teams, so we scan every file of the test type and
    filter by `team:`.
  * A file's test type is the underscore-delimited token before `_tests.yaml`
    (unit, e2e, sanity, stress, perf, integration, l2, device_perf, smoke,
    profiler, sweep, health, demo). `device_perf` and `perf` are distinct.
  * Per entry: skus: { <machine>: { timeout: <minutes>, tier: <n> } }.

Usage:
  query_time_budget.py --team models --testtype unit --machine wh_n150 [--tier 1] [-v]
"""

import argparse
import glob
import os
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTS_DIR = os.path.join(REPO_ROOT, "tests", "pipeline_reorg")
DEFAULT_BUDGET_FILE = os.path.join(REPO_ROOT, ".github", "time_budget.yaml")


def testtype_of_file(filename):
    """Return the test-type token of a `*_tests.yaml` file, or None.

    e.g. models_unit_tests.yaml      -> "unit"
         models_device_perf_tests.yaml -> "device_perf"
         galaxy_perf_tests.yaml      -> "perf"
    """
    base = os.path.basename(filename)
    if not base.endswith("_tests.yaml"):
        return None
    stem = base[: -len("_tests.yaml")]  # e.g. "models_device_perf"
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    # The test type is the trailing token(s). Treat "device_perf" specially so it
    # is not confused with "perf"; everything else is the single last token.
    if len(parts) >= 2 and parts[-2] == "device" and parts[-1] == "perf":
        return "device_perf"
    return parts[-1]


def find_test_files(tests_dir, testtype):
    """All *_tests.yaml files in tests_dir whose test type equals `testtype`."""
    return sorted(f for f in glob.glob(os.path.join(tests_dir, "*_tests.yaml")) if testtype_of_file(f) == testtype)


def collect_allocations(files, team, machine, tier=None):
    """Return (total_minutes, breakdown) for entries matching team/machine.

    breakdown is a list of (file_basename, test_name, timeout, tier) tuples.
    """
    total = 0
    breakdown = []
    for path in files:
        with open(path, "r") as f:
            entries = yaml.safe_load(f) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("team") != team:
                continue
            skus = entry.get("skus") or {}
            sku_cfg = skus.get(machine)
            if not isinstance(sku_cfg, dict) or "timeout" not in sku_cfg:
                continue
            entry_tier = sku_cfg.get("tier")
            if tier is not None and entry_tier != tier:
                continue
            timeout = sku_cfg["timeout"]
            total += timeout
            breakdown.append((os.path.basename(path), entry.get("name", "Unnamed"), timeout, entry_tier))
    return total, breakdown


def lookup_budget(budget_file, team, testtype, machine):
    """Return the declared budget, or None if not present."""
    with open(budget_file, "r") as f:
        budgets = yaml.safe_load(f) or {}
    try:
        return budgets[team][testtype][machine]
    except (KeyError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", required=True, help="Team name, e.g. models, llk, runtime")
    parser.add_argument("--testtype", required=True, help="Test type, e.g. unit, sanity, stress, e2e")
    parser.add_argument("--machine", required=True, help="SKU/machine name, e.g. wh_n150, wh_llmbox")
    parser.add_argument("--tier", type=int, default=None, help="Optional: only sum entries of this tier")
    parser.add_argument("--tests-dir", default=TESTS_DIR, help="Directory of *_tests.yaml files")
    parser.add_argument("--budget-file", default=DEFAULT_BUDGET_FILE, help="Path to time_budget.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-test breakdown")
    args = parser.parse_args()

    files = find_test_files(args.tests_dir, args.testtype)
    if not files:
        print(f"[WARN] No '*_{args.testtype}_tests.yaml' files found in {args.tests_dir}")

    total, breakdown = collect_allocations(files, args.team, args.machine, args.tier)
    budget = lookup_budget(args.budget_file, args.team, args.testtype, args.machine)

    tier_note = f", tier {args.tier}" if args.tier is not None else ""
    print(f"Team '{args.team}' / test type '{args.testtype}' / machine '{args.machine}'{tier_note}")
    print(f"  Files scanned: {len(files)}")
    if args.verbose:
        for fname, name, timeout, etier in breakdown:
            tlabel = f" (tier {etier})" if etier is not None else ""
            print(f"    {timeout:>5} min  {fname}: {name}{tlabel}")
    print(f"  Tests matched: {len(breakdown)}")
    print(f"  Allocated:     {total} min")

    if budget is None:
        print(f"  Budget:        (not declared in {os.path.basename(args.budget_file)})")
        # No budget to compare against; allocation is informational only.
        return 0

    headroom = budget - total
    print(f"  Budget:        {budget} min")
    if headroom >= 0:
        print(f"  [OK] Within budget ({headroom} min headroom).")
        return 0
    print(f"  [OVER] Exceeds budget by {-headroom} min.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
