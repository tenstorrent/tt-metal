# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check that every commit that landed on main on one day and changed an LLK perf
input was measured and compared by the LLK perf merge gate (#63949).

The commits and their diffs come from git, so the audit costs no API request per
commit. Each commit is classified again by its own llk-perf-inputs-changed.sh,
with its own perf_*.py beside it, and matched by SHA to the merge_group run of
merge-gate.yaml that tested it.
"""

import argparse
import concurrent.futures
import datetime as dt
import io
import json
import pathlib
import re
import statistics
import subprocess
import tempfile
import time
import zipfile

CLASSIFIER = ".github/scripts/llk-perf-inputs-changed.sh"
PERF_TESTS = "tt_metal/tt-llk/tests/python_tests"
GATE = "LLK perf merge gate"
ARCHES = ("blackhole", "wormhole")
LEG = re.compile(r"_([a-z]+) group (\d+)/(\d+)")
PR = re.compile(r"\(#(\d+)\)\s*$")
POINTS = re.compile(r"(\d+) points compared")
VERDICTS = ("pass", "regression")
SHORT_SHA = 11


def _gh(*args, tries=4):
    # The jobs endpoint of a merge-gate run (~150 jobs) often answers 502.
    for attempt in range(tries):
        out = subprocess.run(["gh", *args], capture_output=True, text=True)
        if out.returncode == 0 or attempt == tries - 1:
            out.check_returncode()
            return out.stdout
        time.sleep(10 * (attempt + 1))


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout


def landed(repo, ref, day):
    """First-parent commits of ``ref`` committed on ``day`` (UTC), oldest first."""
    start = dt.datetime.combine(day, dt.time(), dt.timezone.utc)
    out = _git(
        repo,
        "log",
        "--first-parent",
        f"--since={(start - dt.timedelta(days=1)).isoformat()}",
        f"--until={(start + dt.timedelta(days=2)).isoformat()}",
        "--format=%H%x09%ct%x09%s",
        ref,
    )
    commits = []
    for line in out.splitlines():
        sha, stamp, subject = line.split("\t", 2)
        if dt.datetime.fromtimestamp(int(stamp), dt.timezone.utc).date() == day:
            pr = PR.search(subject)
            commits.append({"sha": sha, "pr": int(pr.group(1)) if pr else None})
    return commits[::-1]


def needs_gate(repo, sha):
    """The commit's own classifier on its merged diff, with its perf tests present."""
    files = _git(
        repo, "diff", "--name-only", "--diff-filter=ACMRT", f"{sha}^", sha
    ).splitlines()
    wanted = [CLASSIFIER]
    if any(re.fullmatch(rf"{PERF_TESTS}/test_[^/]*\.py", f) for f in files):
        wanted += [
            p
            for p in _git(repo, "ls-tree", "--name-only", sha, f"{PERF_TESTS}/").split()
            if re.search(r"/perf_[^/]*\.py$", p)
        ]
    with tempfile.TemporaryDirectory() as root:
        for path in wanted:
            target = pathlib.Path(root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_git(repo, "show", f"{sha}:{path}"))
        out = subprocess.run(
            ["bash", str(pathlib.Path(root, CLASSIFIER))],
            input="".join(f + "\n" for f in files),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return out == "true"


def queue_runs(repo, day):
    """merge_group runs that tested a queue entry, by head SHA, from the day before to the day after.

    A commit's date is when the queue made it, and its run can start hours later.
    """
    runs = _json_lines(
        _gh(
            "api",
            "--paginate",
            f"repos/{repo}/actions/workflows/merge-gate.yaml/runs?event=merge_group"
            f"&created={day - dt.timedelta(days=1)}..{day + dt.timedelta(days=1)}&per_page=100",
            "--jq",
            ".workflow_runs[] | {id, html_url, head_sha, display_title, created_at} | @json",
        )
    )
    out = {}
    for r in sorted(runs, key=lambda r: r["created_at"]):
        if "Processing" in r["display_title"]:
            out[r["head_sha"]] = r
    return out


def queue_run(repo, sha):
    """The run for one SHA, asked directly before a commit is called unqueued."""
    runs = _json_lines(
        _gh(
            "api",
            f"repos/{repo}/actions/workflows/merge-gate.yaml/runs?event=merge_group&head_sha={sha}",
            "--jq",
            ".workflow_runs[] | {id, html_url, head_sha, display_title, created_at} | @json",
        )
    )
    runs = [r for r in runs if "Processing" in r["display_title"]]
    return max(runs, key=lambda r: r["created_at"]) if runs else None


def run_jobs(repo, run_id):
    return _json_lines(
        _gh(
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&filter=latest",
            "--jq",
            f'.jobs[] | select(.name | startswith("{GATE}")) | {{name, conclusion}} | @json',
        )
    )


def _artifact(repo, artifact_id, cache):
    path = pathlib.Path(cache, f"{artifact_id}.zip")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
            capture_output=True,
            check=True,
        ).stdout
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").rename(path)
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
        return {
            n: z.read(n).decode() for n in z.namelist() if n.endswith((".txt", ".md"))
        }


def gate_outputs(repo, run_id, cache):
    """Per arch: the verdict kind and the points compared, from the gate's artifacts."""
    artifacts = {
        a["name"]: a["id"]
        for a in _json_lines(
            _gh(
                "api",
                f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
                "--paginate",
                "--jq",
                '.artifacts[] | select(.name | startswith("llk-perf-gate-")) | select(.expired | not) | {name, id} | @json',
            )
        )
    }
    out = {}
    for arch in ARCHES:
        kind = points = None
        if f"llk-perf-gate-verdict-{arch}" in artifacts:
            text = "".join(
                _artifact(
                    repo, artifacts[f"llk-perf-gate-verdict-{arch}"], cache
                ).values()
            ).split()
            kind = text[1] if len(text) > 1 else None
        report = None
        if f"llk-perf-gate-report-{arch}" in artifacts and kind in (None, *VERDICTS):
            report = _artifact(repo, artifacts[f"llk-perf-gate-report-{arch}"], cache)
            report = next(iter(report.values()), "")
            match = POINTS.search(report)
            points = int(match.group(1)) if match else None
        # Runs before #63850 (2026-11-03) upload no verdict; read it from the report.
        if kind is None and report:
            kind = "regression" if "REGRESSIONS FOUND" in report else "pass"
        out[arch] = {"kind": kind, "points": points}
    return out


def assess(run, jobs, outputs):
    """The first finding for one commit that needs the gate, or None."""
    if run is None:
        return "no-queue-run", ["no merge_group run tested this SHA"]
    by_name = {j["name"]: j for j in jobs}
    changes = by_name.get(f"{GATE} (changes)")
    measured = [
        j
        for j in jobs
        if j["name"].startswith(f"{GATE} (measure)")
        and j["conclusion"] not in ("skipped", None)
    ]
    if changes is None or changes["conclusion"] != "success" or not measured:
        said = changes["conclusion"] if changes else "no changes job"
        return "skipped", [
            f"changes job {said}" if said != "success" else "changes job said no"
        ]
    failed = []
    for j in jobs:
        arch = next((a for a in ARCHES if j["name"].startswith(f"{GATE} ({a})")), None)
        if (arch or j["name"].startswith(f"{GATE} (measure)")) and j[
            "conclusion"
        ] not in ("success", "skipped"):
            what = f"{arch} compare" if arch else j["name"].split(" / ")[-1]
            failed.append(f"{what.split(' [')[0]} {j['conclusion'] or 'not finished'}")
    detail = []
    for arch in ARCHES:
        legs = {}
        for j in jobs:
            m = LEG.search(j["name"])
            if m and m.group(1) == arch:
                legs[int(m.group(2))] = int(m.group(3))
        total = set(legs.values())
        if len(total) != 1 or set(legs) != set(range(1, max(total) + 1)):
            detail.append(f"{arch}: legs {sorted(legs)} of {sorted(total) or '?'}")
        gate = [j for j in jobs if j["name"].startswith(f"{GATE} ({arch})")]
        kind = outputs.get(arch, {}).get("kind")
        if all(j["conclusion"] in ("skipped", None) for j in gate):
            detail.append(f"{arch}: compare job did not run")
        elif all(j["conclusion"] == "success" for j in gate) and kind not in VERDICTS:
            detail.append(f"{arch}: verdict {kind or 'missing'}")
    if failed:
        return "failed", sorted(failed) + detail
    return ("incomplete", detail) if detail else None


def short_points(commits, ratio):
    """Mark a compare with far fewer points than the day's median for its arch."""
    for arch in ARCHES:
        counts = [
            c["outputs"][arch]["points"]
            for c in commits
            if c.get("outputs", {}).get(arch, {}).get("points")
        ]
        if not counts:
            continue
        floor = ratio * statistics.median(counts)
        for c in commits:
            points = c.get("outputs", {}).get(arch, {}).get("points")
            if c.get("finding") is None and points and points < floor:
                c["finding"] = (
                    "incomplete",
                    [
                        f"{arch}: {points:,} points compared, day median {statistics.median(counts):,.0f}"
                    ],
                )


def audit(repo, git_dir, ref, day, cache, ratio=0.9):
    commits = landed(git_dir, ref, day)
    gated = [c for c in commits if needs_gate(git_dir, c["sha"])]
    runs = queue_runs(repo, day) if gated else {}

    def check(c):
        c["run"], c["outputs"] = None, {}
        try:
            c["run"] = runs.get(c["sha"]) or queue_run(repo, c["sha"])
            jobs = run_jobs(repo, c["run"]["id"]) if c["run"] else []
            if jobs:
                c["outputs"] = gate_outputs(repo, c["run"]["id"], cache)
            c["finding"] = assess(c["run"], jobs, c["outputs"])
        except subprocess.CalledProcessError as e:
            c["finding"] = (
                "not read",
                [((e.stderr or "").strip().splitlines() or [str(e)])[-1][:200]],
            )

    # A run's job list takes ~10 s to page through; ask for several at once.
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(check, gated))
    short_points(gated, ratio)
    return commits, gated


def report(day, commits, gated, most=10):
    found = [c for c in gated if c["finding"]]
    groups = {}
    for c in found:
        kind, detail = c["finding"]
        groups.setdefault((kind, "; ".join(detail)), []).append(
            f"#{c['pr']}" if c["pr"] else c["sha"][:SHORT_SHA]
        )
    problems = [
        f"merge gate: {len(prs)} landed {kind} ({detail}): {', '.join(prs)}"
        for (kind, detail), prs in groups.items()
    ]
    lines = [
        f"{len(commits)} commits landed, {len(gated)} changed LLK perf inputs, "
        f"{len(gated) - len(found)} measured and compared, {len(found)} finding(s)"
    ]
    lines += [
        f"`{c['sha'][:SHORT_SHA]}` {c['finding'][0]}"
        + (f" <{c['run']['html_url']}|run>" if c["run"] else "")
        for c in found[:most]
    ]
    if len(found) > most:
        lines.append(f"and {len(found) - most} more, in the job log")
    return {"name": f"Merge-gate audit, {day}", "problems": problems, "lines": lines}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--git-dir", default=".")
    ap.add_argument("--ref", default="origin/main")
    yesterday = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    ap.add_argument(
        "--day", default=yesterday.isoformat(), help="UTC day the commits landed"
    )
    ap.add_argument("--cache", default=".merge_gate_audit_cache")
    ap.add_argument("--out", default="nightly_checks/merge-gate-audit.json")
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.day)
    commits, gated = audit(a.repo, a.git_dir, a.ref, day, a.cache)
    for c in gated:
        print(
            c["sha"][:SHORT_SHA],
            f"#{c['pr']}",
            c["run"]["html_url"] if c["run"] else "-",
            c["finding"][0] if c["finding"] else "ok",
            "; ".join(c["finding"][1]) if c["finding"] else "",
            json.dumps(c["outputs"]),
        )
    out = report(day, commits, gated)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print("\n".join([out["name"], *out["problems"], *out["lines"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
