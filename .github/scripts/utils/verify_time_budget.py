#!/usr/bin/env python3

"""Verify every test yaml's timeouts against the team time budgets, repo-wide.

This runs ONCE per PR (the verify-time-budgets job in pr-gate.yaml) over every
tests/pipeline_reorg/*.yaml at the same time. Seeing all of them together is the
whole point: several yamls charge the same (team, budget_type, sku) bucket, and
the budget for that bucket is the sum across all of them. Checking one yaml at a
time -- which is what this script used to do, with the budget key passed in as a
CLI argument by each individual workflow -- let every yaml claim the full budget
independently, so a shared bucket could be spent two or three times over.

Bucket resolution
-----------------
Each test entry declares its own bucket coordinates:

    team:        <team>   # which team's allowance to charge
    budget_type: <key>    # which allowance under that team
    skus:
      <sku>:
        timeout: <min>    # minutes charged to (team, budget_type, sku)
        tier: <n>         # optional, see below

The budget is budgets[team][budget_type][sku] in .github/time_budget.yaml.

`budget_type` may also be a LIST, for a tests yaml that one pipeline runs against
one allowance and another pipeline runs against a different one. The timeouts are
charged to every listed allowance, and each must cover them on its own, because
each pipeline spends its own budget when it runs. llk_pr_gate_tests.yaml is the
live case: the PR gate charges llk.pr_gate and sanity-tests charges llk.sanity for
the same tests, deliberately, so that widening one pipeline's pytest markers cannot
silently eat the other's allowance.

Tiers: when a sku entry carries `tier: n`, the key becomes "<budget_type>_tier<n>"
IF the team declares that key, otherwise the plain "<budget_type>" key is used and
the tiers are pooled. That keeps the split a property of the budget file: to break
a pooled budget out per tier, add the tiered keys to time_budget.yaml and nothing
in the test yamls has to change.

Per-test ceiling
----------------
The gates additionally cap how long any ONE test may take, so that a single entry
cannot hold up the whole gate no matter how much budget its team has left. That
applies to the gate budget types below and is checked here rather than passed in
per workflow, which is what the old `per-test-timeout` workflow input did.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "pipeline_reorg")
DEFAULT_BUDGET_FILE = os.path.join(REPO_ROOT, ".github", "time_budget.yaml")

# Budget types that gate a merge, and the ceiling on any single test entry in them.
GATE_BUDGET_TYPES = {"pr_gate", "merge_gate"}
GATE_PER_TEST_CEILING_MINUTES = 15


def resolve_budget_key(budgets, team, budget_type, tier):
    """Bucket key for a sku entry: tiered when the team declares it, else plain."""
    if tier is None:
        return budget_type
    tiered = f"{budget_type}_tier{tier}"
    team_budgets = budgets.get(team)
    if isinstance(team_budgets, dict) and tiered in team_budgets:
        return tiered
    return budget_type


def load_tests(tests_dir):
    """Yield (basename, entries) for each test-list yaml under tests_dir.

    Files whose top level is not a list (e.g. ttsim-skip-list.yaml) are not test
    lists and are skipped.
    """
    for path in sorted(glob.glob(os.path.join(tests_dir, "*.yaml"))):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            continue
        yield os.path.basename(path), data


def collect(tests_dir, budgets, errors):
    """Sum every test's timeouts into its bucket.

    Returns {(team, budget_key, sku): {yaml_basename: minutes}}.
    """
    buckets = defaultdict(lambda: defaultdict(int))

    for basename, entries in load_tests(tests_dir):
        for index, test in enumerate(entries):
            if not isinstance(test, dict):
                errors.append(f"{basename}: entry #{index} is not a mapping.")
                continue

            label = f"{basename}: '{test.get('name', f'entry #{index}')}'"
            missing = [key for key in ("team", "budget_type", "skus") if key not in test]
            if missing:
                errors.append(f"{label} is missing mandatory key(s): {', '.join(missing)}.")
                continue

            team, skus = test["team"], test["skus"]
            if not isinstance(skus, dict) or not skus:
                errors.append(f"{label} has an invalid 'skus' field; expected a non-empty mapping.")
                continue

            # A single key or a list of them; charged to each independently.
            declared = test["budget_type"]
            budget_types = declared if isinstance(declared, list) else [declared]
            if not budget_types or not all(isinstance(b, str) for b in budget_types):
                errors.append(f"{label} has an invalid 'budget_type'; expected a string or list of strings.")
                continue
            gated = [b for b in budget_types if b in GATE_BUDGET_TYPES]

            for sku, config in skus.items():
                if not isinstance(config, dict) or "timeout" not in config:
                    errors.append(f"{label}, SKU '{sku}' is missing 'timeout'.")
                    continue

                timeout = config["timeout"]
                if gated and timeout > GATE_PER_TEST_CEILING_MINUTES:
                    errors.append(
                        f"{label}, SKU '{sku}' has timeout {timeout} min, over the "
                        f"{GATE_PER_TEST_CEILING_MINUTES} min per-test ceiling for "
                        f"{' and '.join(repr(b) for b in gated)}."
                    )

                for budget_type in budget_types:
                    key = (team, resolve_budget_key(budgets, team, budget_type, config.get("tier")), sku)
                    buckets[key][basename] += timeout

    return buckets


def declared_buckets(budgets):
    """Every (team, budget_type, sku) triple that has a budget declared."""
    declared = set()
    for team, budget_types in budgets.items():
        if not isinstance(budget_types, dict):
            continue
        for budget_type, skus in budget_types.items():
            if isinstance(skus, dict):
                declared.update((team, budget_type, sku) for sku in skus)
    return declared


def check(buckets, budgets, errors):
    """Compare each bucket's total against its budget. Returns report rows."""
    rows = []
    for key in sorted(buckets, key=lambda k: (k[0] or "", k[1], k[2])):
        team, budget_type, sku = key
        contributors = buckets[key]
        total = sum(contributors.values())
        try:
            budget = budgets[team][budget_type][sku]
        except (KeyError, TypeError):
            errors.append(
                f"No budget declared for team '{team}', budget_type '{budget_type}', SKU '{sku}' "
                f"(charged {total} min by {', '.join(sorted(contributors))})."
            )
            rows.append((team, budget_type, sku, total, None, contributors))
            continue

        if total > budget:
            errors.append(
                f"Over budget: team '{team}', budget_type '{budget_type}', SKU '{sku}' "
                f"sums to {total} min against a budget of {budget} min (over by {total - budget}). "
                f"Contributors: {', '.join(f'{f} {m} min' for f, m in sorted(contributors.items()))}."
            )
        rows.append((team, budget_type, sku, total, budget, contributors))
    return rows


def render(rows, unused, errors):
    """Write the full allocation table to stdout and the GitHub step summary."""
    lines = ["# Time budget report", ""]
    lines.append(f"{len(rows)} buckets charged. " + ("**FAILED**" if errors else "All within budget."))
    lines.append("")
    lines.append("| Team | Budget type | SKU | Allocated | Budget | Headroom | Test yamls |")
    lines.append("|---|---|---|---|---|---:|---|")
    for team, budget_type, sku, total, budget, contributors in rows:
        headroom = "n/a" if budget is None else str(budget - total)
        flag = " :warning:" if budget is not None and total > budget else ""
        files = ", ".join(f"{f} ({m})" for f, m in sorted(contributors.items(), key=lambda x: -x[1]))
        lines.append(
            f"| {team} | {budget_type} | {sku} | {total} | {'none' if budget is None else budget} "
            f"| {headroom}{flag} | {files} |"
        )

    if unused:
        lines.append("")
        lines.append(f"## Unused budget entries ({len(unused)})")
        lines.append("")
        lines.append("No test charges these, so the capacity they reserve is idle:")
        lines.append("")
        for team, budget_type, sku in sorted(unused):
            lines.append(f"- `{team}` / `{budget_type}` / `{sku}`")

    if errors:
        lines.append("")
        lines.append(f"## Errors ({len(errors)})")
        lines.append("")
        lines.extend(f"- {error}" for error in errors)

    report = "\n".join(lines)
    print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests-dir", default=DEFAULT_TESTS_DIR, help="Directory of test yamls")
    parser.add_argument("--budget-file", default=DEFAULT_BUDGET_FILE, help="Path to time_budget.yaml")
    args = parser.parse_args()

    with open(args.budget_file, "r") as f:
        budgets = yaml.safe_load(f) or {}

    errors = []
    buckets = collect(args.tests_dir, budgets, errors)
    rows = check(buckets, budgets, errors)
    unused = declared_buckets(budgets) - set(buckets)

    render(rows, unused, errors)

    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
