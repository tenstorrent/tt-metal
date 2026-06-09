"""Query allocated test time for a (team, test type, machine) and compare it to
the budget declared in .github/time_budget.yaml.

The companion script verify_time_budget.py checks one tests file against the
budget during CI. This tool answers the inverse question: "How much time does
team X currently allocate for test type Y on machine Z?" by summing the
per-SKU timeouts across the tests files covered by that budget key.

Notes on the data model (see tests/pipeline_reorg/*_tests.yaml):
  * Each entry has a `team:` field. This field -- not the file name -- is the
    authoritative owner. A single file (e.g. galaxy_unit_tests.yaml) mixes
    entries from several teams, so we scan every file of the test type and
    filter by `team:`.
  * A file's test type is the underscore-delimited token before `_tests.yaml`
    (unit, e2e, sanity, stress, perf, integration, l2, device_perf, smoke,
    profiler, sweep, health, demo). `device_perf` and `perf` are distinct.
  * Per entry: skus: { <machine>: { timeout: <minutes>, tier: <n> } }.
  * The models unit/e2e/sweep budgets are split. The plain keys cover non-tiered
    pipelines; the unit_tier<n>/e2e_tier<n>/sweep_tier<n> keys cover
    models_<testtype>_tests.yaml.

It also estimates weekly machine-hours = budget x (cron runs/week) / 60. The
cron frequency is discovered live by parsing .github/workflows at query time
(so it tracks schedule changes): the workflow that runs each contributing test
file is found via its TESTS_YAML_PATH, then the scheduled workflow(s) that
trigger it are read for their cron schedules. Only cron triggers are counted;
manual workflow_dispatch runs are intentionally excluded from the estimate.

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
TIERED_MODEL_TESTTYPES = {"unit", "e2e", "sweep"}
WORKFLOWS_DIR = os.path.join(REPO_ROOT, ".github", "workflows")


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


def uses_tiered_model_budget(team, testtype):
    """Whether this query should use the models-specific tiered budget split."""
    return team == "models" and testtype in TIERED_MODEL_TESTTYPES


def select_files_for_budget(files, team, testtype, tier):
    """Return the files that correspond to the budget key being queried.

    The models unit/e2e/sweep budgets are split: the plain keys cover non-tiered
    pipelines, while the <testtype>_tier<n> keys cover models_<testtype>_tests.yaml.
    """
    if not uses_tiered_model_budget(team, testtype):
        return files

    tiered_filename = f"models_{testtype}_tests.yaml"
    if tier is not None:
        return [f for f in files if os.path.basename(f) == tiered_filename]
    return [f for f in files if os.path.basename(f) != tiered_filename]


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


def lookup_budget(budget_file, team, testtype, machine, tier=None):
    """Return the declared budget, or None if not present."""
    with open(budget_file, "r") as f:
        budgets = yaml.safe_load(f) or {}
    try:
        team_budgets = budgets[team]
        tiered_key = f"{testtype}_tier{tier}"
        if tier is not None and tiered_key in team_budgets:
            return team_budgets[tiered_key][machine]
        return team_budgets[testtype][machine]
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
    parser.add_argument("--testtype", required=True, help="Test type, e.g. unit, sanity, stress, e2e")
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

    matching_files = find_test_files(args.tests_dir, args.testtype)
    files = select_files_for_budget(matching_files, args.team, args.testtype, args.tier)
    if not matching_files:
        print(f"[WARN] No '*_{args.testtype}_tests.yaml' files found in {args.tests_dir}")
    elif not files:
        print(f"[WARN] No files correspond to this budget key after applying model tier split rules")

    total, breakdown = collect_allocations(files, args.team, args.machine, args.tier)
    budget = lookup_budget(args.budget_file, args.team, args.testtype, args.machine, args.tier)

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
