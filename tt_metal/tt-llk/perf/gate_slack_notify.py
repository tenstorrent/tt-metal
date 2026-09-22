# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Turn a gate verdict into a Slack payload, ready for ``chat.postMessage``."""

import argparse
import csv
import json
import os
import shlex


_TOP_N = 5


_NOT_CONFIG = {
    "marker",
    "run_type",
    "current",
    "baseline",
    "delta_pct",
    "delta_cycles",
}


_TEST_KEYS = ("test", "testname", "test_name", "test_module")

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
        k: v
        for k, v in row.items()
        if k not in _NOT_CONFIG and k not in _TEST_KEYS and v not in (None, "")
    }


def _describe(row):
    """One bullet: what moved, by how much, and on which configuration."""
    config = _config_of(row)
    pairs = [f"{k}={v}" for k, v in list(config.items())[:_MAX_CONFIG_PAIRS]]
    if len(config) > _MAX_CONFIG_PAIRS:
        pairs.append("…")
    marker = row.get("marker") or "?"
    run_type = row.get("run_type") or "?"
    pct = row.get("delta_pct") or "?"
    cycles = row.get("delta_cycles") or "?"
    test = next((row[k] for k in _TEST_KEYS if row.get(k)), None)
    what = f"{test} {marker}" if test else marker
    line = f"• {what} {run_type}  +{pct}% (+{cycles} cy)"
    if pairs:
        line += "  — " + " · ".join(pairs)
    return line


def _test_name(rows):
    for row in rows:
        for key in _TEST_KEYS:
            if row.get(key):
                return row[key]
    return None


def _repro_command(rows, arch, baseline_sha):
    script = "tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh"
    test = _test_name(rows) or "<perf_test_module>"
    cmd = [script, arch or "<arch>", test]
    if baseline_sha and baseline_sha != "?":
        cmd += ["--baseline", baseline_sha]
    return " ".join(shlex.quote(part) for part in cmd)


def verdict(*, exit_code, have_baseline, report_written):
    """``(status, reason)`` — ``regressed`` | ``skipped`` | ``clean``."""
    if not have_baseline:
        return "skipped", "No baseline was found, so the gate compared nothing."
    if not report_written:
        return "skipped", "The comparison did not run, so the gate compared nothing."
    return ("regressed", "") if exit_code != 0 else ("clean", "")


def build_text(status, rows, ctx):
    """The Slack message for one verdict. Plain text; Slack renders the links."""
    pr = (
        f"<{ctx['pr_url']}|#{ctx['pr_number']}>"
        if ctx.get("pr_url")
        else f"#{ctx['pr_number']}"
    )
    author = ctx.get("author") or "unknown"
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
        lines.insert(2, f"_{ctx['pr_title']}_")
    if worst:
        lines.append("")
        lines += [_describe(row) for row in worst]
    if len(rows) > _TOP_N:
        lines.append(f"_… and {len(rows) - _TOP_N} more._")
    lines += [
        "",
        f"The full table is in the PR comment. {run_link}",
        "",
        "Reproduce locally:",
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
    ap.add_argument(
        "--notify-on-success",
        action="store_true",
        help="also post when the gate passes (useful while testing the bot)",
    )
    ap.add_argument(
        "--quiet-skipped",
        action="store_true",
        help="do not post when the gate found no baseline",
    )
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
    )
    rows = read_regressions(a.regressions) if status == "regressed" else []

    should_post = (
        status == "regressed"
        or (status == "skipped" and not a.quiet_skipped)
        or (status == "clean" and a.notify_on_success)
    )

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
