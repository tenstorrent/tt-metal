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


def problem(kind, message, **fields):
    """One failure. `message` goes to the log and the annotation; `kind` groups it
    in the PR comment, where `fields` are rendered into something actionable."""
    return {"kind": kind, "message": message, **fields}


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


def collect(tests_dir, budgets, problems):
    """Sum every test's timeouts into its bucket.

    Returns {(team, budget_key, sku): {yaml_basename: minutes}}.
    """
    buckets = defaultdict(lambda: defaultdict(int))

    for basename, entries in load_tests(tests_dir):
        for index, test in enumerate(entries):
            if not isinstance(test, dict):
                problems.append(
                    problem(
                        "not_mapping",
                        f"{basename}: entry #{index} is not a mapping.",
                        yaml=basename,
                        test=f"entry #{index}",
                    )
                )
                continue

            name = test.get("name", f"entry #{index}")
            label = f"{basename}: '{name}'"
            missing = [key for key in ("team", "budget_type", "skus") if key not in test]
            if missing:
                problems.append(
                    problem(
                        "missing_field",
                        f"{label} is missing mandatory key(s): {', '.join(missing)}.",
                        yaml=basename,
                        test=name,
                        missing=missing,
                    )
                )
                continue

            team, skus = test["team"], test["skus"]
            if not isinstance(skus, dict) or not skus:
                problems.append(
                    problem(
                        "bad_skus",
                        f"{label} has an invalid 'skus' field; expected a non-empty mapping.",
                        yaml=basename,
                        test=name,
                    )
                )
                continue

            # A single key or a list of them; charged to each independently.
            declared = test["budget_type"]
            budget_types = declared if isinstance(declared, list) else [declared]
            if not budget_types or not all(isinstance(b, str) for b in budget_types):
                problems.append(
                    problem(
                        "bad_budget_type",
                        f"{label} has an invalid 'budget_type'; expected a string or list of strings.",
                        yaml=basename,
                        test=name,
                        found=repr(declared),
                    )
                )
                continue
            gated = [b for b in budget_types if b in GATE_BUDGET_TYPES]

            for sku, config in skus.items():
                if not isinstance(config, dict) or "timeout" not in config:
                    problems.append(
                        problem(
                            "missing_timeout",
                            f"{label}, SKU '{sku}' is missing 'timeout'.",
                            yaml=basename,
                            test=name,
                            sku=sku,
                        )
                    )
                    continue

                timeout = config["timeout"]
                if gated and timeout > GATE_PER_TEST_CEILING_MINUTES:
                    problems.append(
                        problem(
                            "ceiling",
                            f"{label}, SKU '{sku}' has timeout {timeout} min, over the "
                            f"{GATE_PER_TEST_CEILING_MINUTES} min per-test ceiling for "
                            f"{' and '.join(repr(b) for b in gated)}.",
                            yaml=basename,
                            test=name,
                            sku=sku,
                            timeout=timeout,
                            gated=gated,
                        )
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


def check(buckets, budgets, problems):
    """Compare each bucket's total against its budget. Returns report rows."""
    rows = []
    for key in sorted(buckets, key=lambda k: (k[0] or "", k[1], k[2])):
        team, budget_type, sku = key
        contributors = buckets[key]
        total = sum(contributors.values())
        ranked = dict(sorted(contributors.items(), key=lambda x: -x[1]))
        try:
            budget = budgets[team][budget_type][sku]
        except (KeyError, TypeError):
            problems.append(
                problem(
                    "undeclared",
                    f"No budget declared for team '{team}', budget_type '{budget_type}', SKU '{sku}' "
                    f"(charged {total} min by {', '.join(sorted(contributors))}).",
                    team=team,
                    budget_type=budget_type,
                    sku=sku,
                    allocated=total,
                    contributors=ranked,
                )
            )
            rows.append((team, budget_type, sku, total, None, contributors))
            continue

        if total > budget:
            problems.append(
                problem(
                    "overflow",
                    f"Over budget: team '{team}', budget_type '{budget_type}', SKU '{sku}' "
                    f"sums to {total} min against a budget of {budget} min (over by {total - budget}). "
                    f"Contributors: {', '.join(f'{f} {m} min' for f, m in sorted(contributors.items()))}.",
                    team=team,
                    budget_type=budget_type,
                    sku=sku,
                    allocated=total,
                    budget=budget,
                    over_by=total - budget,
                    contributors=ranked,
                )
            )
        rows.append((team, budget_type, sku, total, budget, contributors))
    return rows


HEADERS = ("Team", "Budget type", "SKU", "Allocated", "Budget", "Headroom", "Test yamls")
# Right-align the three numeric columns; the rest read better left-aligned.
NUMERIC_COLUMNS = {3, 4, 5}


def table_cells(rows):
    """One list of display strings per row, in HEADERS order."""
    cells = []
    for team, budget_type, sku, total, budget, contributors in rows:
        over = budget is not None and total > budget
        headroom = "n/a" if budget is None else str(budget - total)
        cells.append(
            [
                team,
                budget_type,
                sku,
                str(total),
                "none" if budget is None else str(budget),
                headroom + (" !" if over else ""),
                ", ".join(f"{f} ({m})" for f, m in sorted(contributors.items(), key=lambda x: -x[1])),
            ]
        )
    return cells


def render_text(rows, unused, errors):
    """Column-aligned plain text, for reading in the raw job log.

    Padded with spaces rather than tabs: a tab advances to the next 8-column stop,
    so any cell longer than its stop pushes the rest of the row out of alignment.
    """
    cells = table_cells(rows)
    widths = [max(len(h), *(len(row[i]) for row in cells)) if cells else len(h) for i, h in enumerate(HEADERS)]

    def line(values):
        padded = [v.rjust(widths[i]) if i in NUMERIC_COLUMNS else v.ljust(widths[i]) for i, v in enumerate(values)]
        return "  ".join(padded).rstrip()

    header = line(HEADERS)
    rule = len(header)

    def section(title):
        return ["", f"-- {title} ".ljust(rule, "-"), ""]

    out = ["", "=" * rule, "Time budget report", "=" * rule, ""]
    out.append(f"{len(rows)} buckets charged. " + ("FAILED" if errors else "All within budget."))
    out.append("")
    out.append(header)
    out.append("  ".join("-" * w for w in widths))
    out.extend(line(row) for row in cells)

    if unused:
        out += section(f"Unused budget entries ({len(unused)})")
        out.append("No test charges these, so the capacity they reserve is idle:")
        out.extend(f"  {team} / {budget_type} / {sku}" for team, budget_type, sku in sorted(unused))

    if errors:
        out += section(f"Errors ({len(errors)})")
        out.extend(f"  * {error}" for error in errors)

    out.append("")
    return "\n".join(out)


def render_markdown(rows, unused, errors):
    """Markdown, for the GitHub step summary where it renders as a real table."""
    out = ["# Time budget report", ""]
    out.append(f"{len(rows)} buckets charged. " + ("**FAILED**" if errors else "All within budget."))
    out += ["", "| " + " | ".join(HEADERS) + " |", "|---|---|---|---:|---:|---:|---|"]
    for row in table_cells(rows):
        cells = list(row)
        cells[5] = cells[5].replace(" !", " :warning:")
        out.append("| " + " | ".join(cells) + " |")

    if unused:
        out += ["", f"## Unused budget entries ({len(unused)})", ""]
        out.append("No test charges these, so the capacity they reserve is idle:")
        out.append("")
        out.extend(f"- `{team}` / `{budget_type}` / `{sku}`" for team, budget_type, sku in sorted(unused))

    if errors:
        out += ["", f"## Errors ({len(errors)})", ""]
        out.extend(f"- {error}" for error in errors)

    return "\n".join(out)


def render(rows, unused, errors):
    """Aligned text to the job log, markdown to the step summary."""
    print(render_text(rows, unused, errors))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(render_markdown(rows, unused, errors) + "\n")


# Marker the workflow greps for, so repeated pushes update one comment instead
# of stacking a new one on every run. The trailing status= tells the workflow
# whether this body is a failure report or the "now passing" note.
COMMENT_MARKER = "<!-- time-budget-check"


def yaml_permalink(basename):
    """Blob URL for a tests yaml at the commit under test, or None off CI."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (server and repo and sha):
        return None
    return f"{server}/{repo}/blob/{sha}/tests/pipeline_reorg/{basename}"


BUDGET_FILE_LINK = (
    "[`.github/time_budget.yaml`](https://github.com/tenstorrent/tt-metal/blob/main/.github/time_budget.yaml)"
)

# Per-kind heading and the explanation of how to fix it. Order sets comment order.
PROBLEM_KINDS = (
    (
        "undeclared",
        "No budget declared for a bucket",
        "These tests charge a `(team, budget_type, sku)` bucket that has no budget in "
        f"{BUDGET_FILE_LINK}. Either add the SKU under that team's `budget_type`, or point "
        "the test at a `budget_type` the team already has.",
    ),
    (
        "missing_field",
        "Test entry missing a mandatory key",
        "Every entry needs `team`, `budget_type` and `skus`. `budget_type` names which of "
        f"the team's allowances in {BUDGET_FILE_LINK} the test is charged against — the same "
        "key the pipeline that runs it is budgeted under (e.g. `unit`, `merge_gate`, `e2e`).",
    ),
    (
        "ceiling",
        "Test over the gate per-test ceiling",
        f"No single test in a gate may exceed {GATE_PER_TEST_CEILING_MINUTES} min, so that one "
        "entry cannot hold up the whole gate however much budget its team has left. Split the "
        "test, or move it to a non-gate pipeline.",
    ),
    (
        "missing_timeout",
        "SKU entry missing a timeout",
        "Every SKU under `skus` needs a `timeout` in minutes — that is the number charged against the budget.",
    ),
    (
        "bad_budget_type",
        "Invalid `budget_type`",
        "`budget_type` must be a string, or a list of strings when one yaml is run by two "
        "pipelines that charge different allowances.",
    ),
    (
        "bad_skus",
        "Invalid `skus` field",
        "`skus` must be a non-empty mapping of SKU name to its config.",
    ),
    (
        "not_mapping",
        "Entry is not a mapping",
        "Each item in a tests yaml must be a mapping of fields, not a bare scalar or list.",
    ),
)


def yaml_link(basename):
    """Markdown link to a tests yaml, or bare text when the permalink is unavailable."""
    url = yaml_permalink(basename)
    return f"[{basename}]({url})" if url else f"`{basename}`"


def comment_body(problems):
    """Markdown PR comment explaining every failure, grouped by kind."""
    if not problems:
        return "\n".join(
            [
                f"{COMMENT_MARKER} status=pass -->",
                "## :white_check_mark: Time budget check passed",
                "",
                "Every `(team, budget_type, sku)` bucket is within its budget. The full "
                "allocation table is in the job summary.",
            ]
        )

    by_kind = defaultdict(list)
    for item in problems:
        by_kind[item["kind"]].append(item)

    plural = "problem" if len(problems) == 1 else "problems"
    lines = [
        f"{COMMENT_MARKER} status=fail -->",
        "## :rotating_light: Time budget check failed",
        "",
        f"pr-gate's `verify-time-budgets` found {len(problems)} {plural}.",
    ]

    overflows = by_kind.pop("overflow", [])
    if overflows:
        lines += [
            "",
            f"### Budget exceeded ({len(overflows)})",
            "",
            f"Trim the timeouts below, or raise the budget in {BUDGET_FILE_LINK} with a justification in the PR description.",
            "",
            "| Budget bucket | Over by | Allocated | Budget | Test yamls charging it |",
            "|---|---:|---:|---:|---|",
        ]
        for overflow in overflows:
            charged = "<br>".join(
                f"{yaml_link(basename)} ({minutes} min)" for basename, minutes in overflow["contributors"].items()
            )
            lines.append(
                f"| `{overflow['team']}` / `{overflow['budget_type']}` / `{overflow['sku']}` "
                f"| **{overflow['over_by']} min** | {overflow['allocated']} | {overflow['budget']} "
                f"| {charged} |"
            )

    for kind, heading, explanation in PROBLEM_KINDS:
        items = by_kind.pop(kind, [])
        if not items:
            continue
        lines += ["", f"### {heading} ({len(items)})", "", explanation, ""]
        for item in items:
            if kind == "undeclared":
                charged = ", ".join(yaml_link(b) for b in item["contributors"])
                lines.append(
                    f"- `{item['team']}` / `{item['budget_type']}` / `{item['sku']}` — "
                    f"{item['allocated']} min charged by {charged}"
                )
            elif kind == "ceiling":
                lines.append(
                    f"- {yaml_link(item['yaml'])} → **{item['test']}** on `{item['sku']}` — "
                    f"{item['timeout']} min, over the {GATE_PER_TEST_CEILING_MINUTES} min ceiling "
                    f"for {', '.join(f'`{g}`' for g in item['gated'])}"
                )
            elif kind == "missing_field":
                keys = ", ".join(f"`{k}`" for k in item["missing"])
                lines.append(f"- {yaml_link(item['yaml'])} → **{item['test']}** — missing {keys}")
            elif kind == "missing_timeout":
                lines.append(f"- {yaml_link(item['yaml'])} → **{item['test']}** — SKU `{item['sku']}` has no `timeout`")
            elif kind == "bad_budget_type":
                lines.append(f"- {yaml_link(item['yaml'])} → **{item['test']}** — got `{item['found']}`")
            else:
                lines.append(f"- {yaml_link(item['yaml'])} → **{item['test']}**")

    # Anything whose kind predates its entry in PROBLEM_KINDS still gets reported.
    for kind, items in by_kind.items():
        lines += ["", f"### {kind} ({len(items)})", ""]
        lines.extend(f"- {item['message']}" for item in items)

    return "\n".join(lines)


def write_comment(problems, path):
    """Write the PR comment body. Always written, so a fixed PR can clear its comment."""
    with open(path, "w") as f:
        f.write(comment_body(problems) + "\n")
    state = f"{len(problems)} problem(s)" if problems else "pass"
    print(f"\nWrote PR comment body ({state}) to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests-dir", default=DEFAULT_TESTS_DIR, help="Directory of test yamls")
    parser.add_argument("--budget-file", default=DEFAULT_BUDGET_FILE, help="Path to time_budget.yaml")
    parser.add_argument(
        "--comment-file",
        default=None,
        help="Write the PR comment body here. Always written when given: a report of every "
        "problem found, or a short note that the check passed so a PR that fixed its "
        "overspend can clear the stale comment.",
    )
    args = parser.parse_args()

    with open(args.budget_file, "r") as f:
        budgets = yaml.safe_load(f) or {}

    problems = []
    buckets = collect(args.tests_dir, budgets, problems)
    rows = check(buckets, budgets, problems)
    unused = declared_buckets(budgets) - set(buckets)

    render(rows, unused, [item["message"] for item in problems])

    if args.comment_file:
        write_comment(problems, args.comment_file)

    if problems:
        for item in problems:
            print(f"::error::{item['message']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
