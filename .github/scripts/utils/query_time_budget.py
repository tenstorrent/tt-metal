#!/usr/bin/env python3

"""Report how much CI test time a (team, test type, machine[, tier]) uses, and
compare it against the budget declared in .github/time_budget.yaml.

verify_time_budget.py runs in CI to fail a pipeline whose timeouts exceed budget.
This script is the interactive companion: it answers "how much time does team X
allocate for test type Y on machine Z, how does that compare to budget, and how
many machine-hours/week does it cost?"

How it works
------------
1. Allocated time: sum the per-SKU `timeout` values from the relevant
   tests/pipeline_reorg/*_tests.yaml entries, filtered by team, machine and
   (optionally) tier.
2. Budget: look up budgets[team][test type][machine] in time_budget.yaml.
3. Machine-hours/week: budget x (cron runs/week) / 60.

Key conventions
---------------
* Owner is the entry's `team:` field, not the file name. One file may hold
  several teams' entries, so all files of a test type are scanned and filtered.
* A test's "test type" is its own `budget_type:` field. An entry may declare a
  list of them, in which case it is charged to each independently and matches a
  query for any of them.
* The models unit/e2e/sweep budgets are tier-split: the plain key covers the
  non-tiered pipelines, while unit_tier<n>/e2e_tier<n>/sweep_tier<n> cover
  models_<testtype>_tests.yaml. Pass --tier to target a tiered budget.
* Cron frequency is parsed from .github/workflows by following the test
  file -> running workflow -> scheduled caller(s). Only cron schedules count;
  manual workflow_dispatch runs are excluded from the estimate.

Usage
-----
  query_time_budget.py --team <team> --budget-type <type> --machine <sku> [--tier N] [-v]

Example
-------
  $ query_time_budget.py --team models --budget-type unit --machine wh_n150 --tier 1 -v
  Team 'models' / test type 'unit' / machine 'wh_n150', tier 1
    Files scanned: 1
         40 min  models_unit_tests.yaml: Llama 3.1-8B unit tests (tier 1)
          8 min  models_unit_tests.yaml: DeepSeek-V3 module unit tests (host-side) (tier 1)
         10 min  models_unit_tests.yaml: Whisper unit tests (tier 1)
         60 min  models_unit_tests.yaml: TT-DiT common unit tests, shared amongst all models. (tier 1)
    Tests matched: 4
    Allocated:     118 min
    Budget:        118 min
    Scheduled pipelines (cron):
         14 runs/wk  models-t1-unit-tests.yaml
    Est. machine-hours/week: 27.5 h  (118 min x 14 runs/wk / 60)
    Note: estimate uses cron schedules only; manual workflow_dispatch runs are NOT counted.
    [OK] Within budget (0 min headroom).


  $ query_time_budget.py --team runtime --budget-type unit --machine wh_n150_civ2
  Team 'runtime' / test type 'unit' / machine 'wh_n150_civ2'
    Files scanned: 1
    Tests matched: 21
    Allocated:     232 min
    Budget:        232 min
    Scheduled pipelines (cron):
          7 runs/wk  code-coverage.yaml
          7 runs/wk  runtime-unit-tests.yaml
    Est. machine-hours/week: 54.1 h  (232 min x 14 runs/wk / 60)
    Note: estimate uses cron schedules only; manual workflow_dispatch runs are NOT counted.
    [OK] Within budget (0 min headroom).
"""

import argparse
import glob
import os
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTS_DIR = os.path.join(REPO_ROOT, "tests", "pipeline_reorg")
DEFAULT_BUDGET_FILE = os.path.join(REPO_ROOT, ".github", "time_budget.yaml")
WORKFLOWS_DIR = os.path.join(REPO_ROOT, ".github", "workflows")

# Share the bucket-resolution rule with the CI check so the two cannot diverge.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_time_budget import load_tests, resolve_budget_key  # noqa: E402


def target_budget_key(budgets, team, budget_type, tier):
    """The budget key this query is asking about."""
    return resolve_budget_key(budgets, team, budget_type, tier)


def collect_allocations(tests_dir, budgets, team, budget_type, machine, tier=None):
    """Return (total_minutes, breakdown) for entries charged to this bucket.

    An entry contributes when its team matches, it has a timeout on `machine`,
    and one of its declared `budget_type` values resolves to the same budget key
    the query is targeting. breakdown is a list of
    (file_basename, test_name, timeout, tier) tuples.
    """
    wanted = target_budget_key(budgets, team, budget_type, tier)
    total = 0
    breakdown = []
    for basename, entries in load_tests(tests_dir):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("team") != team:
                continue

            declared = entry.get("budget_type")
            budget_types = declared if isinstance(declared, list) else [declared]
            if budget_type not in budget_types:
                continue

            sku_cfg = (entry.get("skus") or {}).get(machine)
            if not isinstance(sku_cfg, dict) or "timeout" not in sku_cfg:
                continue

            entry_tier = sku_cfg.get("tier")
            if tier is not None and entry_tier != tier:
                continue
            if resolve_budget_key(budgets, team, budget_type, entry_tier) != wanted:
                continue

            timeout = sku_cfg["timeout"]
            total += timeout
            breakdown.append((basename, entry.get("name", "Unnamed"), timeout, entry_tier))
    return total, breakdown


def lookup_budget(budgets, team, budget_type, machine, tier=None):
    """Return the declared budget, or None if not present."""
    try:
        return budgets[team][target_budget_key(budgets, team, budget_type, tier)][machine]
    except (KeyError, TypeError):
        return None


def _expand_cron_field(field, lo, hi):
    """Expand a single cron field, supporting '*', comma lists, ranges and steps."""
    values = set()
    for part in field.split(","):
        step = 1
        token = part
        if "/" in token:
            token, step_str = token.split("/", 1)
            step = int(step_str)
        if token == "*":
            start, end = lo, hi
        elif "-" in token:
            a, b = token.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(token)
        values.update(range(start, end + 1, step))
    return values


def cron_runs_per_week(cron):
    """Approximate how many times a 5-field cron expression fires per week.

    Handles '*', comma lists, ranges and steps in the minute, hour and
    day-of-week fields. Day-of-month / month restrictions are not modelled
    (assumed '*'); such schedules are rare for these pipelines.
    """
    fields = cron.split()
    if len(fields) != 5:
        return 0
    minute, hour, _dom, _month, dow = fields
    per_day = len(_expand_cron_field(minute, 0, 59)) * len(_expand_cron_field(hour, 0, 23))
    if dow.strip() == "*":
        days = 7
    else:
        # cron allows 7 as Sunday; normalise to 0-6 before counting distinct days.
        days = len({0 if d == 7 else d for d in _expand_cron_field(dow, 0, 7)})
    return per_day * days


def _workflow_call(job):
    """Return (impl_basename, tier_value) for a reusable-workflow job, else None."""
    uses = job.get("uses")
    if not isinstance(uses, str) or ".github/workflows/" not in uses:
        return None
    impl = uses.split(".github/workflows/", 1)[1].split("@", 1)[0].strip()
    tier = (job.get("with") or {}).get("tier")
    return os.path.basename(impl), tier


def build_workflow_index(workflows_dir):
    """Parse every workflow once, capturing schedule / test-file / reuse metadata.

    Returns a list of dicts: {name, crons, tests_yaml (set), calls [(impl, tier)]}.
    """
    index = []
    for path in sorted(glob.glob(os.path.join(workflows_dir, "*.y*ml"))):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            continue
        if not isinstance(data, dict):
            continue

        # PyYAML parses the `on:` trigger key as the boolean True (YAML 1.1).
        triggers = data.get("on", data.get(True)) or {}
        crons = []
        if isinstance(triggers, dict) and isinstance(triggers.get("schedule"), list):
            crons = [s["cron"] for s in triggers["schedule"] if isinstance(s, dict) and "cron" in s]

        jobs = data.get("jobs") or {}
        env_dicts = [data["env"]] if isinstance(data.get("env"), dict) else []
        calls = []
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            if isinstance(job.get("env"), dict):
                env_dicts.append(job["env"])
            call = _workflow_call(job)
            if call:
                calls.append(call)

        tests_yaml = set()
        for env in env_dicts:
            value = env.get("TESTS_YAML_PATH")
            if isinstance(value, str) and value.endswith("_tests.yaml"):
                tests_yaml.add(os.path.basename(value))

        index.append({"name": os.path.basename(path), "crons": crons, "tests_yaml": tests_yaml, "calls": calls})
    return index


def discover_scheduled_runs(workflow_index, test_file, tier):
    """Map a test file to {scheduled_workflow_basename: runs_per_week}.

    Walks the reuse graph: a test file is run by the impl whose TESTS_YAML_PATH
    references it, and that impl is triggered by scheduled caller workflows. When
    a tier is queried, callers that pass a non-matching `tier:` input are skipped.
    """
    runner_names = {w["name"] for w in workflow_index if test_file in w["tests_yaml"]}
    runs = {}
    for w in workflow_index:
        if not w["crons"]:
            continue
        weekly = sum(cron_runs_per_week(c) for c in w["crons"])
        # A workflow that directly declares the test file and is itself scheduled.
        if w["name"] in runner_names:
            runs[w["name"]] = weekly
        # A scheduled workflow that calls the impl which runs the test file.
        for impl, tier_val in w["calls"]:
            if impl not in runner_names:
                continue
            if tier is not None and tier_val is not None and str(tier_val) != str(tier):
                continue
            runs[w["name"]] = weekly
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", required=True, help="Team name, e.g. models, llk, runtime")
    # --budget-type matches the tests yaml field; --testtype is the original spelling.
    parser.add_argument(
        "--budget-type",
        "--testtype",
        dest="testtype",
        required=True,
        help="Budget type, e.g. unit, sanity, stress, e2e",
    )
    parser.add_argument("--machine", required=True, help="SKU/machine name, e.g. wh_n150, wh_llmbox")
    parser.add_argument(
        "--tier", type=int, choices=(1, 2, 3), default=None, help="Optional: only sum entries of this tier"
    )
    parser.add_argument("--tests-dir", default=TESTS_DIR, help="Directory of *_tests.yaml files")
    parser.add_argument("--budget-file", default=DEFAULT_BUDGET_FILE, help="Path to time_budget.yaml")
    parser.add_argument(
        "--workflows-dir", default=WORKFLOWS_DIR, help="Directory of workflow YAMLs (for cron frequency)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-test breakdown")
    args = parser.parse_args()

    with open(args.budget_file, "r") as f:
        budgets = yaml.safe_load(f) or {}

    total, breakdown = collect_allocations(args.tests_dir, budgets, args.team, args.testtype, args.machine, args.tier)
    budget = lookup_budget(budgets, args.team, args.testtype, args.machine, args.tier)
    if not breakdown:
        print(f"[WARN] No tests declare budget_type '{args.testtype}' for team '{args.team}' on '{args.machine}'")

    tier_note = f", tier {args.tier}" if args.tier is not None else ""
    print(f"Team '{args.team}' / test type '{args.testtype}' / machine '{args.machine}'{tier_note}")
    print(f"  Files scanned: {len({fname for fname, _, _, _ in breakdown})}")
    if args.verbose:
        for fname, name, timeout, etier in breakdown:
            tlabel = f" (tier {etier})" if etier is not None else ""
            print(f"    {timeout:>5} min  {fname}: {name}{tlabel}")
    print(f"  Tests matched: {len(breakdown)}")
    print(f"  Allocated:     {total} min")

    if budget is None:
        print(f"  Budget:        (not declared in {os.path.basename(args.budget_file)})")
    else:
        print(f"  Budget:        {budget} min")

    # --- Estimated weekly machine-hours (cron-scheduled runs only) ---
    workflow_index = build_workflow_index(args.workflows_dir)
    contributing_files = sorted({fname for fname, _, _, _ in breakdown})
    runs_by_workflow = {}
    for fname in contributing_files:
        for workflow, weekly in discover_scheduled_runs(workflow_index, fname, args.tier).items():
            runs_by_workflow[workflow] = weekly
    total_runs = sum(runs_by_workflow.values())

    print("  Scheduled pipelines (cron):")
    if runs_by_workflow:
        for workflow in sorted(runs_by_workflow):
            print(f"      {runs_by_workflow[workflow]:>3} runs/wk  {workflow}")
    else:
        print("      (none found -- manual workflow_dispatch only)")

    if budget is not None:
        machine_hours = budget * total_runs / 60.0
        print(f"  Est. machine-hours/week: {machine_hours:.1f} h  ({budget} min x {total_runs} runs/wk / 60)")
    else:
        print("  Est. machine-hours/week: n/a (no budget declared)")
    print("  Note: estimate uses cron schedules only; manual workflow_dispatch runs are NOT counted.")

    if budget is None:
        # No budget to compare against; allocation is informational only.
        return 0
    headroom = budget - total
    if headroom >= 0:
        print(f"  [OK] Within budget ({headroom} min headroom).")
        return 0
    print(f"  [OVER] Exceeds budget by {-headroom} min.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
