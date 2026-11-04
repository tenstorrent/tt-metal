# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Turn a gate verdict into a Slack payload, ready for ``chat.postMessage``."""

import argparse
import csv
import html
import json
import os
import re
import shlex

_TOP_N = 5
_SHORT_SHA = 11

_BROADCAST = re.compile(r"<!(here|channel|everyone)(\||>)")


def _escape(value):
    """Make untrusted text safe for a Slack mrkdwn field."""
    text = _BROADCAST.sub(r"@\1 ", str(value))
    return html.escape(text, quote=False)


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


def _findings(rows):
    """Points that differ only in marker or run type are one finding."""
    groups = {}
    for row in rows:
        key = (row.get(MODULE_KEY) or "", tuple(sorted(_config_of(row).items())))
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _describe_finding(group, varying):
    """The worst point of a finding, and how many points it stands for."""
    line = _describe(group[0], varying)
    if len(group) > 1:
        markers = ", ".join(sorted({r.get("marker") or "?" for r in group}))
        run_types = ", ".join(sorted({r.get("run_type") or "?" for r in group}))
        line += f"  (×{len(group)} points: {markers} · {run_types})"
    return line


# gate.py's verdict status -> the Slack status; every other status is a skip.
_STATUS = {"regression": "regressed", "pass": "clean"}


def from_verdict(v):
    """``(status, ctx)`` of a verdict-<arch>.json; ``None`` (no file) is a skip."""
    if v is None:
        return "skipped", {
            "reason": "Its gate job wrote no verdict, so the gate compared nothing."
        }
    p, base = v["policy"], v.get("baseline") or {}
    note = ""
    if base.get("stale"):
        note = (
            f"Stale baseline: none at {base.get('expected_commit', '?')[:_SHORT_SHA]} yet, so this "
            f"compares with {(base.get('commit_sha') or '?')[:_SHORT_SHA]} ({base.get('run_ts', '?')})."
        )
    ctx = {
        "arch": v["arch"],
        "run_types": ",".join(p["run_types"]),
        "baseline_sha": base.get("commit_sha", ""),
        "gate_name": p["title"],
        "mode": "speed of light" if p["speed_of_light"] else "non speed-of-light",
        "threshold": str(p["threshold"]),
        "min_cycles": str(p["min_cycles"]),
        "module_thresholds": ",".join(f"{m}={t}" for m, t in p["modules"].items()),
        "reason": v["reason"],
        "note": note,
    }
    return _STATUS.get(v["status"], "skipped"), ctx


def _subject(ctx):
    """Headline, byline and scope: a PR when there is one, else the run."""
    number = str(ctx.get("pr_number") or "").strip()
    if number and number != "?":
        pr = f"<{ctx['pr_url']}|#{number}>" if ctx.get("pr_url") else f"#{number}"
        commit = (ctx.get("commit") or "")[:_SHORT_SHA]
        author = _escape(ctx.get("author") or "unknown")
        return (
            f"on PR {pr}",
            f"Author: `{author}`" + (f"  ·  `{commit}`" if commit else ""),
            f"{ctx.get('arch') or '?'} · {ctx.get('run_types') or '?'}",
        )
    bits = [
        b
        for b in (
            ctx.get("arch"),
            ctx.get("mode"),
            (ctx.get("commit") or "")[:_SHORT_SHA],
        )
        if b
    ]
    return (
        f"({_escape(ctx.get('trigger') or 'nightly')})",
        " · ".join(bits) or "scheduled run",
        ctx.get("run_types") or "?",
    )


def _report_line(ctx, run_link):
    """Where the full table is: this gate's own summary, not the whole run page."""
    links = []
    if ctx.get("report_url"):
        links.append(f"<{ctx['report_url']}|full table>")
    if ctx.get("artifact_url"):
        links.append(f"<{ctx['artifact_url']}|download the report and CSVs>")
    if not links:
        return f"The full table is in the report attached to the {run_link}."
    return "Report: " + " · ".join(links)


def build_text(status, rows, ctx):
    """The Slack message for one verdict. Plain text; Slack renders the links."""
    subject, byline, where = _subject(ctx)
    gate = _escape(ctx.get("gate_name") or "LLK perf gate")
    note = [f":hourglass: {_escape(ctx['note'])}"] if ctx.get("note") else []
    run_link = f"<{ctx['run_url']}|gate run>" if ctx.get("run_url") else "gate run"

    if status == "skipped":
        return "\n".join(
            [
                f":warning: *{gate} SKIPPED* {subject}",
                f"{byline}  ·  {where}",
                *note,
                "",
                _escape(ctx.get("reason") or "The gate compared nothing."),
                "*A green check here does not mean anything was checked.*",
                "",
                f"{run_link}",
            ]
        )

    if status == "clean":
        return "\n".join(
            [
                f":white_check_mark: *{gate} passed* {subject}",
                f"{byline}  ·  {where}",
                *note,
                "",
                f"No point regressed. {run_link}",
            ]
        )

    lines = [
        f":rotating_light: *{gate}: regression* {subject}",
        f"{byline}  ·  {where}",
        *note,
        "",
    ]
    if ctx.get("pr_title"):
        lines.insert(2, f"_{_escape(ctx['pr_title'])}_")
    lines += regression_body(rows, ctx)
    return "\n".join(lines)


def regression_body(rows, ctx, annotate=None):
    """Count, worst findings, report links and a repro command for one arch.

    ``annotate(row)`` may return one more line for a finding's worst point."""
    run_link = f"<{ctx['run_url']}|gate run>" if ctx.get("run_url") else "gate run"
    findings = _findings(rows)
    worst = findings[:_TOP_N]
    lines = [f"*{len(rows)} point(s) regressed, in {len(findings)} finding(s).*"]
    if worst:
        lines.append("")
        varying = _varying_keys([g[0] for g in worst])
        for group in worst:
            lines.append(_describe_finding(group, varying))
            extra = annotate(group[0]) if annotate else ""
            if extra:
                lines.append(f"      ↳ {extra}")
    if len(findings) > _TOP_N:
        lines.append(f"_… and {len(findings) - _TOP_N} more finding(s)._")
    lines += [
        "",
        _report_line(ctx, run_link),
        "",
        "Reproduce the worst one locally:",
        f"```{_repro_command(rows, ctx.get('arch'), ctx.get('baseline_sha'))}```",
    ]
    return lines


def _emit_outputs(**values):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the gate's Slack payload.")
    ap.add_argument("--verdict", required=True, help="the gate's verdict-<arch>.json")
    ap.add_argument("--regressions", help="path to <stem>.regressions.csv")
    ap.add_argument("--channel", required=True, help="Slack channel id")
    ap.add_argument("--pr-number", default="?")
    ap.add_argument("--pr-url", default="")
    ap.add_argument("--pr-title", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--run-url", default="")
    ap.add_argument(
        "--report-url", default="", help="this job's summary on the run page"
    )
    ap.add_argument("--artifact-url", default="", help="the uploaded gate report")
    ap.add_argument("--commit", default="")
    ap.add_argument(
        "--trigger", default="nightly", help="what started a run with no PR"
    )
    ap.add_argument(
        "--post-pass",
        default="true",
        help="false: post nothing when the gate passes; regressions and skips still post",
    )
    ap.add_argument(
        "--part-out",
        help="also write the verdict as JSON, for gate_slack_combine.py",
    )
    ap.add_argument("--out", default="slack_payload.json")
    a = ap.parse_args(argv)

    try:
        with open(a.verdict) as fh:
            v = json.load(fh)
    except (OSError, ValueError):
        v = None
    status, ctx = from_verdict(v)
    rows = read_regressions(a.regressions) if status == "regressed" else []

    should_post = status != "clean" or a.post_pass.strip().lower() != "false"

    ctx.update(
        pr_number=a.pr_number,
        pr_url=a.pr_url,
        pr_title=a.pr_title,
        author=a.author,
        commit=a.commit,
        run_url=a.run_url,
        report_url=a.report_url,
        artifact_url=a.artifact_url,
        trigger=a.trigger,
    )
    text = build_text(status, rows, ctx)

    if a.part_out:
        with open(a.part_out, "w") as fh:
            json.dump(
                {
                    "status": status,
                    "should_post": should_post,
                    "rows": rows,
                    "ctx": ctx,
                },
                fh,
            )

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
