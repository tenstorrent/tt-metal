# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Find the commit that made LLK perf unstable across CI cards.

The question is not "did this commit get slower". It is "does this commit give
the same numbers twice". So every measurement compares **two CI runs of the same
commit** against each other. Nothing is ever compared across commits, which is
why a month of CSV schema drift cannot confuse the result.

Each CI run lands on fresh ephemeral cards, so a point that disagrees between the
two runs is one the hardware does not reproduce. That is the signal.

    # one commit, two runs, report the fires
    perf_bisect.py measure 80ec10ab611

    # drive the whole search between a known-good and a known-bad commit
    perf_bisect.py bisect --good 80ec10ab611 --bad 143e28aa7a6

`measure` is idempotent per commit: results land in --state (default
`perf_bisect_state.json`) and are reused, so an interrupted bisect resumes
without re-measuring anything.

Requires `gh`, authenticated, with push access. Measuring a commit pushes a
`perf-bisect/<short-sha>` branch, because workflow_dispatch needs a ref. Use
`--cleanup` to delete those when you are done.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

REPO = "tenstorrent/tt-metal"
WORKFLOW = "llk-perf.yaml"
BRANCH_PREFIX = "perf-bisect/"

# The gate's rule. Kept here so the bisect's verdict and the gate's verdict are
# the same verdict.
THRESHOLD = 0.02
MIN_CYCLES = 30.0

# Every perf-relevant path. A kernel's instruction placement can change from any
# of these, and the CI image is built from the tree, so the toolchain is covered
# by bisecting the tree itself.
PERF_PATHS = [
    "tt_metal/tt-llk",
    "tt_metal/hw/ckernels",
    "tt_metal/hw/inc",
]

POLL_SECONDS = 60
RUN_TIMEOUT_SECONDS = 3 * 60 * 60


def sh(*args, check=True, capture=True):
    r = subprocess.run(args, text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{r.stderr or r.stdout}")
    return (r.stdout or "").strip()


def git(*args, **kw):
    return sh("git", *args, **kw)


def short(sha):
    return sha[:11]


# --- state ------------------------------------------------------------------


def load_state(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(path, state):
    pathlib.Path(path).write_text(json.dumps(state, indent=2, sort_keys=True))


# --- dispatching ------------------------------------------------------------


def dispatch_inputs(sha):
    """The dispatch inputs this commit's workflow actually declares.

    The form has grown over time — `upload-to-warehouse` and `pipeline` are
    recent — so passing today's set to an older commit is rejected.
    """
    import yaml

    text = git("show", f"{sha}:.github/workflows/{WORKFLOW}")
    spec = yaml.safe_load(text)
    on = spec.get(True) or spec.get("on") or {}
    declared = set((on.get("workflow_dispatch") or {}).get("inputs") or {})

    wanted = {
        "architecture": "blackhole",
        "speed-of-light": "false",
        "upload-to-warehouse": "false",
        "pipeline": "",
    }
    return {k: v for k, v in wanted.items() if k in declared and v != ""}


def push_branch(sha):
    branch = BRANCH_PREFIX + short(sha)
    git("push", "--force", f"git@github.com:{REPO}.git", f"{sha}:refs/heads/{branch}")
    return branch


def runs_on(branch):
    out = sh(
        "gh",
        "run",
        "list",
        "--repo",
        REPO,
        "--workflow",
        WORKFLOW,
        "--branch",
        branch,
        "--limit",
        "20",
        "--json",
        "databaseId,status,conclusion,createdAt",
    )
    return json.loads(out or "[]")


def start_runs(sha, branch, count):
    """Dispatch `count` runs and return their ids, newest last."""
    before = {r["databaseId"] for r in runs_on(branch)}
    inputs = dispatch_inputs(sha)
    print(f"  dispatch inputs: {inputs}")
    for i in range(count):
        args = ["gh", "workflow", "run", WORKFLOW, "--repo", REPO, "--ref", branch]
        for k, v in inputs.items():
            args += ["-f", f"{k}={v}"]
        sh(*args)
        print(f"  dispatched run {i + 1}/{count}")
        time.sleep(10)  # so the two runs get distinct queue positions

    deadline = time.time() + 600
    while time.time() < deadline:
        new = [r for r in runs_on(branch) if r["databaseId"] not in before]
        if len(new) >= count:
            ids = sorted(r["databaseId"] for r in new)
            print(f"  run ids: {ids}")
            return ids
        time.sleep(15)
    raise RuntimeError(f"only {len(new)} of {count} runs appeared for {branch}")


def wait_for(run_ids):
    deadline = time.time() + RUN_TIMEOUT_SECONDS
    pending = set(run_ids)
    while pending and time.time() < deadline:
        for rid in sorted(pending):
            info = json.loads(
                sh(
                    "gh",
                    "run",
                    "view",
                    str(rid),
                    "--repo",
                    REPO,
                    "--json",
                    "status,conclusion",
                )
            )
            if info["status"] == "completed":
                print(f"  run {rid}: {info['conclusion']}")
                pending.discard(rid)
        if pending:
            time.sleep(POLL_SECONDS)
    if pending:
        raise RuntimeError(f"timed out waiting for {sorted(pending)}")


# --- artifacts --------------------------------------------------------------


def fetch_perf_data(run_id, dest):
    """Download a run's perf artifacts, flattening the two layouts it may use.

    Before 2026-08-26 a shard published `perf_data-<arch>-<group>.zip`; after, a
    directory tree. Both end up as CSVs under `dest`.
    """
    dest = pathlib.Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    sh(
        "gh",
        "run",
        "download",
        str(run_id),
        "--repo",
        REPO,
        "--pattern",
        "perf-data-*",
        "-D",
        str(dest),
    )

    for zpath in list(dest.rglob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(zpath.parent / zpath.stem)
        zpath.unlink()

    csvs = [p for p in dest.rglob("*.csv") if not p.name.endswith(".post.csv")]
    print(f"  run {run_id}: {len(csvs)} CSV(s)")
    return csvs


# --- the verdict ------------------------------------------------------------


def compare(module, left, right, out_dir):
    """Run the gate's own compare over every run type, fires broken out by type."""
    import pandas as pd

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(module),
            "--baseline",
            f"{left}/**/*.csv",
            "--current",
            f"{right}/**/*.csv",
            "--threshold",
            str(THRESHOLD),
            "--min-cycles",
            str(MIN_CYCLES),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    points = out_dir / "report.points.csv"
    if not points.exists():
        raise RuntimeError(f"no points written; see {report}")

    df = pd.read_csv(points, low_memory=False)
    fired = (df["delta_pct"].abs() > THRESHOLD * 100) & (
        df["delta_cycles"].abs() > MIN_CYCLES
    )
    by_type = {}
    for run_type, group in df.groupby("run_type"):
        mask = fired.loc[group.index]
        by_type[str(run_type)] = {
            "points": int(len(group)),
            "fires": int(mask.sum()),
        }
    return {"points": int(len(df)), "fires": int(fired.sum()), "by_run_type": by_type}


def compare_module(ref, work_dir="."):
    """The compare script, taken from one fixed ref so the rule never varies.

    Written under the work directory, never the tree, so a bisect run leaves no
    stray file next to the sources it is measuring.
    """
    target = pathlib.Path(work_dir) / "compare_module.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        "tt_metal/tt-llk/perf/regression_compare.py",
        "tt_metal/tt-llk/.claude/scripts/perf_regression_compare.py",
    ):
        try:
            target.write_text(git("show", f"{ref}:{path}"))
            return target
        except RuntimeError:
            continue
    raise RuntimeError(f"no compare module at {ref}")


# --- one commit -------------------------------------------------------------


def measure(sha, args, state):
    key = short(sha)
    if key in state and not args.refresh:
        print(f"{key}: cached -> {state[key]['fires']} fires")
        return state[key]

    subject = git("log", "-1", "--format=%s", sha)[:70]
    print(
        f"\n=== {key}  {git('log', '-1', '--format=%ad', '--date=short', sha)}  {subject}"
    )

    branch = push_branch(sha)
    run_ids = start_runs(sha, branch, args.runs)
    wait_for(run_ids)

    work = pathlib.Path(args.work_dir) / key
    sides = []
    for i, rid in enumerate(run_ids[: args.runs]):
        side = work / f"run{i}"
        fetch_perf_data(rid, side)
        sides.append(side)

    module = compare_module(args.compare_ref, args.work_dir)
    result = compare(module, sides[0], sides[1], work / "compare")
    result.update(
        commit=sha,
        subject=subject,
        run_ids=run_ids,
        dispatch_inputs=dispatch_inputs(sha),
    )

    verdict = "UNSTABLE" if result["fires"] else "stable"
    print(f"  -> {verdict}: {result['fires']} fire(s) over {result['points']} points")
    for rt, v in sorted(result["by_run_type"].items()):
        if v["fires"]:
            print(f"       {rt}: {v['fires']} / {v['points']}")

    state[key] = result
    save_state(args.state, state)
    return result


# --- the search -------------------------------------------------------------


def candidates(good, bad):
    """Perf-relevant commits in (good, bad], oldest first."""
    out = git("rev-list", "--reverse", f"{good}..{bad}", "--", *PERF_PATHS)
    return out.split() if out else []


def bisect(args, state):
    commits = candidates(args.good, args.bad)
    print(f"{len(commits)} perf-relevant commit(s) between the endpoints")

    good_res = measure(args.good, args, state)
    if good_res["fires"]:
        print("\nThe good endpoint is ALSO unstable. Nothing to bisect: the cause is")
        print("not a commit in this range. Suspect the measurement setup instead —")
        print("every CI run lands on different cards, which single-card baselines")
        print("never exercised.")
        return
    bad_res = measure(args.bad, args, state)
    if not bad_res["fires"]:
        print("\nThe bad endpoint is stable. Nothing to bisect.")
        return

    lo, hi = 0, len(commits)  # instability entered in commits[lo:hi]
    while lo < hi:
        mid = (lo + hi) // 2
        res = measure(commits[mid], args, state)
        if res["fires"]:
            hi = mid
        else:
            lo = mid + 1
        print(f"  remaining window: {hi - lo} commit(s)")

    if lo >= len(commits):
        print("\nNo commit in the range reproduces it.")
        return
    culprit = commits[lo]
    print(f"\nFIRST UNSTABLE COMMIT: {short(culprit)}")
    print(git("log", "-1", "--format=%H%n%ad  %an%n%s", "--date=short", culprit))


def cleanup():
    out = sh(
        "gh",
        "api",
        f"repos/{REPO}/branches?per_page=100",
        "--jq",
        f'.[].name | select(startswith("{BRANCH_PREFIX}"))',
    )
    for branch in out.split():
        git("push", f"git@github.com:{REPO}.git", "--delete", branch, check=False)
        print(f"  deleted {branch}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("measure", "bisect", "cleanup"))
    ap.add_argument("commit", nargs="?", help="commit, for `measure`")
    ap.add_argument("--good", help="last commit known to be stable")
    ap.add_argument("--bad", help="first commit known to be unstable")
    ap.add_argument("--runs", type=int, default=2, help="CI runs per commit")
    ap.add_argument("--state", default="perf_bisect_state.json")
    ap.add_argument("--work-dir", default="perf_bisect_work")
    ap.add_argument(
        "--compare-ref",
        default="nstojictt/llk-perf-gate-slack-notify",
        help="ref to take the compare module from; one fixed rule for every point",
    )
    ap.add_argument("--refresh", action="store_true", help="re-measure cached commits")
    a = ap.parse_args(argv)

    if a.action == "cleanup":
        return cleanup()

    state = load_state(a.state)
    if a.action == "measure":
        if not a.commit:
            ap.error("measure needs a commit")
        measure(git("rev-parse", a.commit), a, state)
    else:
        if not (a.good and a.bad):
            ap.error("bisect needs --good and --bad")
        a.good, a.bad = git("rev-parse", a.good), git("rev-parse", a.bad)
        bisect(a, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
