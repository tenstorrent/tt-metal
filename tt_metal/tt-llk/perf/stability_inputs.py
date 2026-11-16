# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""List the PRs merged in a range of main that touched paths which can change LLK perf stability."""

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess

from nightly_health import _gh, _json_lines, run_jobs

T = "tt_metal/tt-llk/tests"
PATHS = (
    "tt_metal/sfpi-version",
    "tt_metal/sfpi-info.sh",
    f"{T}/python_tests/pytest.ini",
    f"{T}/python_tests/helpers/llk_pytest_plugin.py",
    f"{T}/python_tests/helpers/test_config.py",
    f"{T}/python_tests/helpers/device.py",
    f"{T}/python_tests/helpers/perf",
    f"{T}/helpers/include/counters.h",
    f"{T}/helpers/include/profiler.h",
    f"{T}/helpers/include/barrier.h",
    f"{T}/helpers/include/boot.h",
    f"{T}/helpers/include/perf.h",
    f"{T}/helpers/src/brisc.cpp",
    f"{T}/helpers/src/trisc.cpp",
    f"{T}/helpers/ld",
    f"{T}/run_llk_perf_wormhole.sh",
    f"{T}/run_llk_perf_blackhole.sh",
    f"{T}/run_llk_perf_placement.sh",
    f"{T}/run_llk_perf_merge_stage2.sh",
    f"{T}/perf_split_costs.json",
    f"{T}/perf_split_costs_sol.json",
    f"{T}/requirements.txt",
)
NIGHTLY = "llk-perf-gate.yaml"
MERGE_GATE = ".github/workflows/merge-gate.yaml"
PR = re.compile(r"\s*\(#(\d+)\)$")


def nightly_range(repo, day):
    """The heads of the non-SoL nightly of ``day`` and of the one before it that started."""
    runs = _json_lines(
        _gh(
            "api",
            f"repos/{repo}/actions/workflows/{NIGHTLY}/runs?event=schedule&created=<={day}&per_page=10",
            "--jq",
            '.workflow_runs[] | select(.conclusion != "startup_failure") | {head_sha, created_at} | @json',
        )
    )
    if len(runs) < 2 or not runs[0]["created_at"].startswith(day):
        return None
    return runs[1]["head_sha"], runs[0]["head_sha"]


def range_commits(repo, base, head):
    """Commits of base..head, oldest first."""
    return _json_lines(
        _gh(
            "api",
            "--paginate",
            f"repos/{repo}/compare/{base}...{head}?per_page=100",
            "--jq",
            '.commits[] | {sha, date: .commit.committer.date, title: (.commit.message | split("\\n")[0])} | @json',
        )
    )


def touching(repo, path, head, since):
    return set(
        _gh(
            "api",
            "--paginate",
            f"repos/{repo}/commits?sha={head}&path={path}&since={since}&per_page=100",
            "--jq",
            ".[].sha",
        ).split()
    )


def _jobs(repo, run_id):
    for attempt in range(3):
        try:
            return run_jobs(repo, run_id)
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise


def gate_ran(repo, sha):
    """True when a merge-queue run of the merge gate compared this commit's LLK perf, None when unknown."""
    try:
        return _gate_ran(repo, sha)
    except subprocess.CalledProcessError:
        return None


def _gate_ran(repo, sha):
    runs = _gh(
        "api",
        f"repos/{repo}/actions/runs?head_sha={sha}&event=merge_group&per_page=100",
        "--jq",
        f'.workflow_runs[] | select(.path == "{MERGE_GATE}") | .id',
    ).split()
    for run_id in runs:
        if any(
            j["name"].startswith("LLK perf merge gate (")
            and "Compare" in j["name"]
            and j["conclusion"] in ("success", "failure")
            for j in _jobs(repo, run_id)
        ):
            return True
    return False


def report(commits, hits, gated, base, head):
    """``hits``: sha -> paths it touched; ``gated``: sha -> whether the perf gate ran."""
    span = f"`{base[:11]}..{head[:11]}`"
    lines, problems = [], []
    for c in commits:
        if c["sha"] not in hits:
            continue
        pr = PR.search(c["title"])
        name = f"#{pr.group(1)}" if pr else c["sha"][:11]
        title = PR.sub("", c["title"])
        paths = ", ".join(p.rsplit("/", 1)[-1] for p in hits[c["sha"]])
        unknown = " (gate not checked)" if gated.get(c["sha"]) is None else ""
        lines.append(f"• {name} {title} ({paths}){unknown}")
        if gated.get(c["sha"]) is False:
            problems.append(
                f"{name} changed {paths} and merged without the LLK perf merge gate"
            )
    head_line = (
        f"{len(lines)} merge(s) in {span} can change stability; check S1/S2 today:"
        if lines
        else f"no stability-sensitive merges in {span}"
    )
    return {
        "name": "Stability-sensitive merges",
        "problems": problems,
        "lines": [head_line, *lines],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--range", help="BASE..HEAD (default: the last two nightlies)")
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument("--out", default="nightly_checks/stability_inputs.json")
    a = ap.parse_args(argv)
    span = a.range.split("..") if a.range else nightly_range(a.repo, a.date)
    if span is None:
        out = {
            "name": "Stability-sensitive merges",
            "problems": [],
            "lines": [f"no two non-SoL nightlies up to {a.date}; nothing to check"],
        }
    else:
        base, head = span
        commits = range_commits(a.repo, base, head)
        shas = {c["sha"] for c in commits}
        since = min((c["date"] for c in commits), default=None)
        hits = {}
        for path in PATHS if since else ():
            for sha in touching(a.repo, path, head, since) & shas:
                hits.setdefault(sha, []).append(path)
        gated = {sha: gate_ran(a.repo, sha) for sha in hits}
        out = report(commits, hits, gated, base, head)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("\n".join(out["problems"] + out["lines"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
