# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A daily Slack summary of the PR gate, so passes stay visible without one message each.

Run it by path: ``tt_metal/tt-llk`` is not an importable package.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile

WORKFLOW = ".github/workflows/llk-perf-gate.yaml"
STATUSES = (("clean", "passed"), ("regressed", "regressed"), ("skipped", "skipped"))


def _gh(*args):
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


def collect(repo, since):
    """One ``{pr, arch, status, run_url}`` per gate job of each PR run since ``since``."""
    workflow = _gh(
        "api",
        "--paginate",
        f"repos/{repo}/actions/workflows?per_page=100",
        "--jq",
        f'.workflows[] | select(.path == "{WORKFLOW}") | .id',
    ).split()[0]
    lines = _gh(
        "api",
        "--paginate",
        f"repos/{repo}/actions/workflows/{workflow}/runs?event=pull_request"
        f"&created=>={since:%Y-%m-%dT%H:%M:%SZ}&per_page=100",
        "--jq",
        '.workflow_runs[] | select(.conclusion != "skipped")'
        " | {id, html_url, pr: (.pull_requests[0].number // null)} | @json",
    )
    runs = [json.loads(line) for line in lines.splitlines() if line.strip()]
    out = []
    for run in runs:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _gh(
                    "run",
                    "download",
                    str(run["id"]),
                    "-R",
                    repo,
                    "-D",
                    tmp,
                    "-p",
                    "llk-perf-gate-status-*",
                )
            except subprocess.CalledProcessError:
                continue  # every job skipped or cancelled: nothing compared
            for root, _, files in os.walk(tmp):
                for name in files:
                    with open(os.path.join(root, name)) as fh:
                        status = json.load(fh)
                    out.append(
                        {
                            **status,
                            "pr": status.get("pr") or run["pr"],
                            "run_url": run["html_url"],
                        }
                    )
    return out


def build_text(entries, hours):
    """The summary: counts per arch, and the PRs whose gate needs a look."""
    head = f":bar_chart: *LLK perf PR gate, last {hours} h*"
    if not entries:
        return f"{head}: no PR gate run compared anything."
    prs = sorted({str(e["pr"]) for e in entries if e.get("pr")}, key=int)
    lines = [f"{head}: {len(entries)} verdict(s) on {len(prs)} PR(s)."]
    for arch in sorted({e.get("arch") or "?" for e in entries}):
        mine = [e for e in entries if (e.get("arch") or "?") == arch]
        counts = ", ".join(
            f"{sum(e.get('status') == s for e in mine)} {word}" for s, word in STATUSES
        )
        lines.append(f"• {arch}: {counts}")
    passed = sorted(
        {str(e["pr"]) for e in entries if e.get("status") == "clean" and e.get("pr")},
        key=int,
    )
    if passed:
        lines.append("Passed, posted only here: " + ", ".join(f"#{p}" for p in passed))
    lines.append("Regressions and skips posted their own message.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--out", default="digest_payload.json")
    a = ap.parse_args(argv)
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=a.hours)
    text = build_text(collect(a.repo, since), a.hours)
    with open(a.out, "w") as fh:
        json.dump({"channel": a.channel, "text": text}, fh)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
