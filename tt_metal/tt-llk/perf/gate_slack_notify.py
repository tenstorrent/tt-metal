# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Turn a gate verdict into a Slack payload, ready for ``chat.postMessage``."""

import argparse
import csv
import json
import os
import re
import shlex


_TOP_N = 5

_BROADCAST = re.compile(r"<!(here|channel|everyone)(\||>)")


def _escape(value):
    """Make untrusted text safe for a Slack mrkdwn field."""
    text = _BROADCAST.sub(r"@\1 ", str(value))
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


MODULE_KEY = "test_module"

_NOT_CONFIG = {
    "marker",
    MODULE_KEY,
    "run_type",
    "current",
    "baseline",
    "delta_pct",
    "delta_cycles",
}


_MAX_CONFIG_PAIRS = 4


def read_regressions(path):
    """Rows of the regressions CSV, worst first. Empty when the file is absent."""
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _config_of(row):
    """The sweep configuration of a point, without the test name."""
    return {
        k: v for k, v in row.items() if k not in _NOT_CONFIG and v not in (None, "")
    }


def _varying_keys(rows):
    """Config keys that differ across the rows shown; those tell them apart."""
    keys = {k for row in rows for k in _config_of(row)}
    return {k for k in keys if len({row.get(k) for row in rows}) > 1}


def _pct(value):
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "?"


def _cycles(value):
    try:
        return f"{float(value):+,.0f} cy"
    except (TypeError, ValueError):
        return "? cy"


def _describe(row, varying=()):
    """One bullet: what moved, by how much, and on which configuration."""
    config = _config_of(row)
    ordered = [k for k in config if k in varying] + [
        k for k in config if k not in varying
    ]
    pairs = [f"{_escape(k)}={_escape(config[k])}" for k in ordered[:_MAX_CONFIG_PAIRS]]
    if len(config) > _MAX_CONFIG_PAIRS:
        pairs.append("…")
    marker = row.get("marker") or "?"
    run_type = row.get("run_type") or "?"
    module = row.get(MODULE_KEY) or ""
    what = f"{module} {marker}" if module else marker
    line = (
        f"• {what} {run_type}  {_pct(row.get('delta_pct'))} "
        f"({_cycles(row.get('delta_cycles'))})"
    )
    if pairs:
        line += "  — " + " · ".join(pairs)
    return line


def _repro_command(rows, arch, baseline_sha):
    script = "tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh"
    module = next((r[MODULE_KEY] for r in rows if r.get(MODULE_KEY)), None)
    cmd = [script, arch or "<arch>", (module or "<perf_test_module>").split("+")[0]]
    if baseline_sha and baseline_sha != "?":
        cmd += ["--baseline", baseline_sha]
    return " ".join(shlex.quote(part) for part in cmd)


def verdict(*, exit_code, have_baseline, report_written, comparison_finished=True):
    """``(status, reason)`` — ``regressed`` | ``skipped`` | ``clean``."""
    if not have_baseline:
        return "skipped", "No baseline was found, so the gate compared nothing."
    if not report_written:
        return "skipped", "The comparison did not run, so the gate compared nothing."
    if not comparison_finished:
        return "skipped", (
            "The comparison started but did not finish, so its " "verdict is unknown."
        )
    return ("regressed", "") if exit_code != 0 else ("clean", "")


def build_text(status, rows, ctx):
    """The Slack message for one verdict. Plain text; Slack renders the links."""
    pr = (
        f"<{ctx['pr_url']}|#{ctx['pr_number']}>"
        if ctx.get("pr_url")
        else f"#{ctx['pr_number']}"
    )
    author = _escape(ctx.get("author") or "unknown")
    where = f"{ctx.get('arch') or '?'} · {ctx.get('run_types') or '?'}"
    run_link = f"<{ctx['run_url']}|gate run>" if ctx.get("run_url") else "gate run"

    if status == "skipped":
        return "\n".join(
            [
                f":warning: *LLK perf gate SKIPPED* on PR {pr}",
                f"Author: `{author}`",
                "",
                ctx.get("reason") or "The gate compared nothing.",
                "*A green check here does not mean the PR is clean.*",
                "",
                f"{run_link}",
            ]
        )

    if status == "clean":
        return "\n".join(
            [
                f":white_check_mark: *LLK perf gate passed* on PR {pr}",
                f"Author: `{author}`  ·  {where}",
                "",
                f"No point regressed. {run_link}",
            ]
        )

    worst = rows[:_TOP_N]
    lines = [
        f":rotating_light: *LLK perf gate: regression* on PR {pr}",
        f"Author: `{author}`  ·  {where}",
        "",
        f"*{len(rows)} point(s) regressed.*",
    ]
    if ctx.get("pr_title"):
        lines.insert(2, f"_{_escape(ctx['pr_title'])}_")
    if worst:
        lines.append("")
        varying = _varying_keys(worst)
        lines += [_describe(row, varying) for row in worst]
    if len(rows) > _TOP_N:
        lines.append(f"_… and {len(rows) - _TOP_N} more._")
    lines += [
        "",
        f"The full table is in the PR comment. {run_link}",
        "",
        "Reproduce the worst one locally:",
        f"```{_repro_command(rows, ctx.get('arch'), ctx.get('baseline_sha'))}```",
    ]
    return "\n".join(lines)


def _emit_outputs(**values):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the gate's Slack payload.")
    ap.add_argument("--regressions", help="path to <stem>.regressions.csv")
    ap.add_argument("--report", help="path to the Markdown report the gate wrote")
    ap.add_argument("--have-baseline", help="path to have_baseline.txt")
    ap.add_argument("--exit-code", type=int, default=0)
    ap.add_argument("--comparison-finished", default="true")
    ap.add_argument("--channel", required=True, help="Slack channel id")
    ap.add_argument("--pr-number", default="?")
    ap.add_argument("--pr-url", default="")
    ap.add_argument("--pr-title", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--arch", default="")
    ap.add_argument("--run-types", default="")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--baseline-sha", default="")
    ap.add_argument("--out", default="slack_payload.json")
    a = ap.parse_args(argv)

    have_baseline = False
    if a.have_baseline and os.path.exists(a.have_baseline):
        with open(a.have_baseline) as fh:
            have_baseline = fh.read().strip() == "true"

    report_written = bool(a.report) and os.path.exists(a.report)
    status, reason = verdict(
        exit_code=a.exit_code,
        have_baseline=have_baseline,
        report_written=report_written,
        comparison_finished=a.comparison_finished.strip().lower() != "false",
    )
    rows = read_regressions(a.regressions) if status == "regressed" else []

    should_post = True

    ctx = {
        "pr_number": a.pr_number,
        "pr_url": a.pr_url,
        "pr_title": a.pr_title,
        "author": a.author,
        "arch": a.arch,
        "run_types": a.run_types,
        "run_url": a.run_url,
        "baseline_sha": a.baseline_sha,
        "reason": reason,
    }
    text = build_text(status, rows, ctx)

    with open(a.out, "w") as fh:
        json.dump({"channel": a.channel, "text": text}, fh)

    _emit_outputs(
        status=status,
        should_post=str(should_post).lower(),
        regression_count=len(rows),
    )
    print(f"verdict={status} regressions={len(rows)} should_post={should_post}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
