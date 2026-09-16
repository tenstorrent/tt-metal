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
import os
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

# Below this, a difference between two commits is background, not a change.
MIN_SEPARATION = 50

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


RUNNER_SCRIPTS = [
    "tt_metal/tt-llk/tests/run_llk_perf_blackhole.sh",
    "tt_metal/tt-llk/tests/run_llk_perf_wormhole.sh",
]


def force_non_sol(sha):
    """A commit on top of `sha` whose perf runners measure with SoL off.

    Speed of light cannot be turned off from a dispatch on older commits: before
    2026-08-25 the runner hardcodes `--speed-of-light`, and the SPEED_OF_LIGHT
    env knob that today's input drives did not exist. Bisecting without this
    patch measures SoL at one end and non-SoL at the other, which is a mode
    change, not a code change.

    Returns ``(commit, {path: mechanism})``.
    """
    env = dict(os.environ, GIT_INDEX_FILE=os.path.abspath(".perf_bisect_index"))
    if os.path.exists(env["GIT_INDEX_FILE"]):
        os.remove(env["GIT_INDEX_FILE"])
    subprocess.run(["git", "read-tree", sha], env=env, check=True, capture_output=True)

    how = {}
    for path in RUNNER_SCRIPTS:
        try:
            body = git("show", f"{sha}:{path}")
        except RuntimeError:
            continue  # the script postdates this commit
        if "SPEED_OF_LIGHT:-true" in body:
            patched = body.replace("SPEED_OF_LIGHT:-true", "SPEED_OF_LIGHT:-false")
            how[path] = "env-default"
        elif "--speed-of-light" in body:
            patched = body.replace(" --speed-of-light", "")
            how[path] = "flag-removed"
        else:
            how[path] = "already-off"
            continue
        if "--speed-of-light" in patched.replace(
            "SPEED_OF_LIGHT_ARGS=(--speed-of-light)", ""
        ):
            raise RuntimeError(f"{path}: a --speed-of-light survived the patch")

        r = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            input=patched + ("" if body.endswith("\n") else ""),
            text=True,
            capture_output=True,
            check=True,
        )
        blob = r.stdout.strip()
        mode = git("ls-tree", sha, "--", path).split()[0]
        subprocess.run(
            ["git", "update-index", "--cacheinfo", f"{mode},{blob},{path}"],
            env=env,
            check=True,
            capture_output=True,
        )

    tree = subprocess.run(
        ["git", "write-tree"], env=env, check=True, capture_output=True, text=True
    ).stdout.strip()
    commit = subprocess.run(
        [
            "git",
            "commit-tree",
            tree,
            "-p",
            sha,
            "-m",
            "bisect: measure with speed of light off",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        # commit-tree needs an identity even for a throwaway commit
        # (the bisect branches are deleted by `cleanup`).
    ).stdout.strip()
    os.remove(env["GIT_INDEX_FILE"])
    return commit, how


def push_branch(sha, index):
    """One branch per run, because the workflow cancels its own concurrency group.

    llk-perf.yaml sets `group: <workflow>-<arch>-<github.ref>` with
    cancel-in-progress. Two dispatches of the same commit on the same branch
    therefore share a group, and the second kills the first. Different branch,
    different group, and the two runs proceed in parallel.
    """
    branch = f"{BRANCH_PREFIX}{short(sha)}-r{index}"
    head, _ = force_non_sol(sha)
    git("push", "--force", f"git@github.com:{REPO}.git", f"{head}:refs/heads/{branch}")
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


def start_runs(sha, count):
    """Dispatch one run per branch and return their ids."""
    inputs = dispatch_inputs(sha)
    print(f"  dispatch inputs: {inputs}")
    ids = []
    for i in range(1, count + 1):
        branch = push_branch(sha, i)
        before = {r["databaseId"] for r in runs_on(branch)}
        args = ["gh", "workflow", "run", WORKFLOW, "--repo", REPO, "--ref", branch]
        for k, v in inputs.items():
            args += ["-f", f"{k}={v}"]
        sh(*args)

        deadline = time.time() + 300
        while time.time() < deadline:
            new = [r for r in runs_on(branch) if r["databaseId"] not in before]
            if new:
                rid = new[0]["databaseId"]
                ids.append(rid)
                print(f"  run {i}/{count}: {rid} on {branch}")
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"no run appeared for {branch}")
    return ids


def run_conclusion(run_id):
    return json.loads(
        sh("gh", "run", "view", str(run_id), "--repo", REPO, "--json", "conclusion")
    )["conclusion"]


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


def assert_non_sol(*roots):
    """Refuse to draw a verdict on runs that measured speed of light.

    The setting is recorded per row, so the data itself says what was measured.
    Reading it back is the only check that survives a workflow whose inputs mean
    something different at one end of the range than the other.
    """
    import pandas as pd

    seen = set()
    for root in roots:
        for f in pathlib.Path(root).rglob("*.csv"):
            if f.name.endswith(".post.csv"):
                continue
            df = pd.read_csv(f, low_memory=False)
            if "speed_of_light" not in df.columns:
                raise RuntimeError(f"{f}: no speed_of_light column to check")
            seen.update(df["speed_of_light"].astype(str).unique())
    if seen != {"False"}:
        raise RuntimeError(
            f"runs measured speed_of_light={sorted(seen)}, expected False only. "
            "The non-SoL patch did not take at this commit."
        )
    print("  speed_of_light: False in every row")


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

    run_ids = [int(r) for r in args.use_runs.split(",")] if args.use_runs else None
    if run_ids:
        print(f"  reusing run ids: {run_ids}")
    else:
        run_ids = start_runs(sha, args.runs)
    wait_for(run_ids)

    bad = [r for r in run_ids if run_conclusion(r) != "success"]
    if bad:
        raise RuntimeError(
            f"run(s) {bad} did not succeed; a measurement needs two clean runs"
        )

    work = pathlib.Path(args.work_dir) / key
    sides = []
    for i, rid in enumerate(run_ids[: args.runs]):
        side = work / f"run{i}"
        fetch_perf_data(rid, side)
        sides.append(side)

    assert_non_sol(*sides)
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


# The run types that were clean at the good endpoint: the full pipeline and the
# three stages measured in isolation. L1_CONGESTION is deliberately outside it —
# it fired 13 times at the good endpoint against L1_TO_L1's 0, so it is the
# background this search has to see past, not part of the signal.
CORE_RUN_TYPES = ("L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE")


def resolve_signal(signal):
    """`core`, `total`, or a comma-separated list of run types."""
    if signal == "total":
        return None  # every run type
    if signal == "core":
        return CORE_RUN_TYPES
    return tuple(t.strip() for t in signal.split(",") if t.strip())


def signal_fires(result, signal):
    """Fires summed over the run types the search is tracking."""
    wanted = resolve_signal(signal)
    if wanted is None:
        return result["fires"]
    return sum(v["fires"] for k, v in result["by_run_type"].items() if k in wanted)


def candidates(good, bad):
    """Perf-relevant commits in (good, bad], oldest first."""
    out = git("rev-list", "--reverse", f"{good}..{bad}", "--", *PERF_PATHS)
    return out.split() if out else []


def bisect(args, state):
    commits = candidates(args.good, args.bad)
    print(f"{len(commits)} perf-relevant commit(s) between the endpoints")

    good_res = measure(args.good, args, state)
    bad_res = measure(args.bad, args, state)
    lo_fires = signal_fires(good_res, args.signal)
    hi_fires = signal_fires(bad_res, args.signal)
    tracked = resolve_signal(args.signal)
    print(f"\ntracking: {', '.join(tracked) if tracked else 'every run type'}")
    print(f"{args.signal} fires: good={lo_fires}  bad={hi_fires}")

    # A handful of fires is the background every commit carries; the change being
    # hunted is orders of magnitude larger. So the cut sits on a log scale between
    # the two endpoints rather than at "more than zero".
    if hi_fires < max(10 * (lo_fires + 1), MIN_SEPARATION):
        print(
            f"\nThe endpoints are not far enough apart on {args.signal} "
            f"({lo_fires} vs {hi_fires}). Nothing to bisect — pick a different "
            "signal, or endpoints that actually differ."
        )
        return
    threshold = max(int(((lo_fires + 1) * hi_fires) ** 0.5), MIN_SEPARATION // 2)
    print(f"calling a commit unstable at >= {threshold} {args.signal} fire(s)\n")

    lo, hi = 0, len(commits)  # instability entered in commits[lo:hi]
    while lo < hi:
        mid = (lo + hi) // 2
        res = measure(commits[mid], args, state)
        if signal_fires(res, args.signal) >= threshold:
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
    ap.add_argument(
        "--use-runs",
        help="comma-separated run ids to use instead of dispatching, so a run "
        "that already happened is not wasted",
    )
    ap.add_argument(
        "--signal",
        default="core",
        help="which run types drive the search: 'core' (L1_TO_L1 plus the three "
        "isolate modes, all clean at the good endpoint), 'total' (adds "
        "L1_CONGESTION, which fires everywhere), or a comma-separated list",
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
