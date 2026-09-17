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


def dispatch_inputs(sha, arch="blackhole"):
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
        "architecture": arch,
        "speed-of-light": "false",
        "upload-to-warehouse": "false",
        "pipeline": "",
    }
    return {k: v for k, v in wanted.items() if k in declared and v != ""}


RUNNER_SCRIPTS = [
    "tt_metal/tt-llk/tests/run_llk_perf_blackhole.sh",
    "tt_metal/tt-llk/tests/run_llk_perf_wormhole.sh",
]


PYTEST_INI = "tt_metal/tt-llk/tests/python_tests/pytest.ini"
PERF_CORE = "tt_metal/tt-llk/tests/python_tests/helpers/perf/core.py"
PROFILER = "tt_metal/tt-llk/tests/python_tests/helpers/profiler.py"
WIDE_SCHEMA = "tt_metal/tt-llk/tests/python_tests/helpers/perf/wide_schema.py"


def patch_maxschedchunk(body, value):
    """Set xdist's scheduling chunk, which decides each worker's test sequence."""
    import re

    new, n = re.subn(r"--maxschedchunk=\d+", f"--maxschedchunk={value}", body)
    if not n:
        raise RuntimeError("no --maxschedchunk in pytest.ini to patch")
    return new


def patch_run_count(sha, count, env):
    """Measure each point `count` times and also record the minimum.

    Interference from neighbouring cores can only add cycles to a measurement,
    so min() over several executions is the aggregation that should survive it
    where mean() does not.

    Three edits across two files. core.run() takes its repeat count from the
    environment so the runner can set it without touching every call site;
    _stats_timings gains min() in BOTH the aggregation and the column names it
    builds by hand, which have to stay the same length or pandas raises.
    """
    edits = {
        PERF_CORE: [
            (
                "def run(self, perf_report: PerfReport, run_count=1):",
                "def run(self, perf_report: PerfReport, run_count=None):\n"
                '        run_count = run_count or int(os.environ.get("PERF_RUN_COUNT", "1"))',
            )
        ],
        PROFILER: [
            ('.agg(["mean", "std"])', '.agg(["mean", "std", "min"])'),
            ("for stat in (MEAN, STD)]", 'for stat in (MEAN, STD, "min")]'),
        ],
        # Rule 1 of the perf infra: a CSV column that is not in DB_SCHEMA is
        # dropped, and the run fails the schema gate. Declare the min() columns
        # at both sites that enumerate the timing statistics.
        WIDE_SCHEMA: [
            ("    for kind in (MEAN, STD)\n", '    for kind in (MEAN, STD, "min")\n'),
            (
                "    for base in (metric, stat_column(metric, MEAN), stat_column(metric, STD))",
                "    for base in (\n"
                "        metric,\n"
                "        stat_column(metric, MEAN),\n"
                "        stat_column(metric, STD),\n"
                '        stat_column(metric, "min"),\n'
                "    )",
            ),
        ],
    }

    for path, changes in edits.items():
        body = git("show", f"{sha}:{path}")
        for old, new in changes:
            if old not in body:
                raise RuntimeError(f"{path}: cannot find {old[:40]!r}")
            body = body.replace(old, new, 1)
        blob = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            input=body,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        mode = git("ls-tree", sha, "--", path).split()[0]
        subprocess.run(
            ["git", "update-index", "--cacheinfo", f"{mode},{blob},{path}"],
            env=env,
            check=True,
            capture_output=True,
        )
    return f"run_count={count}, min() recorded"


def apply_commit_files(sha, apply_ref, env):
    """Put `apply_ref`'s version of the files it changed onto `sha`'s tree.

    A cherry-pick without the merge machinery, and safe because it refuses to
    guess: every file must be identical in `sha` and in `apply_ref`'s parent, so
    substituting the blob reproduces the change exactly. A file that moved
    underneath the change is reported rather than silently reverted.
    """
    parent = git("rev-parse", f"{apply_ref}^")
    files = [
        f for f in git("show", apply_ref, "--name-only", "--format=").split("\n") if f
    ]
    drifted = [
        f
        for f in files
        if git("rev-parse", f"{sha}:{f}") != git("rev-parse", f"{parent}:{f}")
    ]
    if drifted:
        raise RuntimeError(
            f"{short(apply_ref)} cannot be applied cleanly to {short(sha)}; these "
            f"files changed underneath it: {drifted}"
        )
    for f in files:
        blob = git("rev-parse", f"{apply_ref}:{f}")
        mode = git("ls-tree", apply_ref, "--", f).split()[0]
        subprocess.run(
            ["git", "update-index", "--cacheinfo", f"{mode},{blob},{f}"],
            env=env,
            check=True,
            capture_output=True,
        )
    return f"{len(files)} file(s) from {short(apply_ref)}"


def patch_runner(
    body,
    *,
    workers=None,
    test_filter=None,
    split_into=None,
    slice_group=1,
    run_count=None,
):
    """Reshape one perf runner script for a controlled experiment.

    `workers` sets -n on both passes; 1 means a single xdist worker, so a single
    Tensix with no neighbours running concurrently. `test_filter` adds -k, and
    `split_into` replaces the shard split so group 1 is a small slice of what the
    filter selected — the other groups exit immediately, leaving exactly one card
    doing exactly one sequence.
    """
    import re

    if run_count is not None:
        body = body.replace(
            "mkdir -p perf_data",
            f"export PERF_RUN_COUNT={run_count}\nmkdir -p perf_data",
            1,
        )
    if workers is not None:
        body = re.sub(r"-n \d+", f"-n {workers}", body)
    if test_filter is not None:
        body = body.replace(
            '-m "perf and not accuracy"',
            f'-m "perf and not accuracy" -k "{test_filter}"',
        )
    if split_into is not None:
        body = body.replace(
            '--splits "$N_GROUPS" --group "$GROUP"',
            f"--splits {split_into} --group {slice_group}",
        )
        marker = "mkdir -p perf_data"
        guard = (
            'if [ "${GROUP}" != "1" ]; then\n'
            '  echo "experiment: only group 1 measures; this group exits."\n'
            "  exit 0\n"
            "fi\n" + marker
        )
        body = body.replace(marker, guard, 1)
    return body


def force_non_sol(
    sha, maxschedchunk=None, apply_ref=None, runner_opts=None, run_count=None
):
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
    if run_count is None and runner_opts:
        run_count = runner_opts.get("run_count")
    if run_count is not None:
        how["run_count"] = patch_run_count(sha, run_count, env)
    if apply_ref:
        how["apply"] = apply_commit_files(sha, apply_ref, env)
    if maxschedchunk is not None:
        body = git("show", f"{sha}:{PYTEST_INI}")
        patched = patch_maxschedchunk(body, maxschedchunk)
        r = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            input=patched,
            text=True,
            capture_output=True,
            check=True,
        )
        mode = git("ls-tree", sha, "--", PYTEST_INI).split()[0]
        subprocess.run(
            [
                "git",
                "update-index",
                "--cacheinfo",
                f"{mode},{r.stdout.strip()},{PYTEST_INI}",
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        how[PYTEST_INI] = f"maxschedchunk={maxschedchunk}"

    for path in RUNNER_SCRIPTS:
        try:
            body = git("show", f"{sha}:{path}")
        except RuntimeError:
            continue  # the script postdates this commit
        if runner_opts:
            body = patch_runner(body, **runner_opts)
            how[path + ":runner"] = ", ".join(
                f"{k}={v}" for k, v in runner_opts.items() if v is not None
            )
        if "SPEED_OF_LIGHT:-true" in body:
            patched = body.replace("SPEED_OF_LIGHT:-true", "SPEED_OF_LIGHT:-false")
            how[path] = "env-default"
        elif "--speed-of-light" in body:
            patched = body.replace(" --speed-of-light", "")
            how[path] = "flag-removed"
        elif runner_opts:
            patched = body
            how[path] = "sol-already-off"
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


def runner_opts_of(args):
    opts = {
        "workers": getattr(args, "workers", None),
        "test_filter": getattr(args, "test_filter", None),
        "split_into": getattr(args, "split_into", None),
        "run_count": getattr(args, "run_count", None),
    }
    if not any(v is not None for v in opts.values()):
        return None
    opts["slice_group"] = getattr(args, "slice_group", None) or 1
    return opts


def variant_key(sha, args):
    """Cache key. A variant must not overwrite the plain measurement of a commit."""
    key = short(sha)
    if getattr(args, "apply", None):
        key += f"+{short(git('rev-parse', args.apply))}"
    if getattr(args, "maxschedchunk", None) is not None:
        key += f"+chunk{args.maxschedchunk}"
    arch = getattr(args, "arch", None)
    if arch and arch != "blackhole":
        key += f"+{arch[:2]}"
    for name, tag in (
        ("workers", "n"),
        ("split_into", "s"),
        ("slice_group", "g"),
        ("run_count", "rc"),
        ("test_filter", "k"),
    ):
        v = getattr(args, name, None)
        if v is not None:
            v = str(v).replace("/", "_").replace(" ", "")[:20]
            key += f"+{tag}{v}"
    return key


def push_branch(
    sha, index, maxschedchunk=None, apply_ref=None, runner_opts=None, arch=None
):
    """One branch per run, because the workflow cancels its own concurrency group.

    llk-perf.yaml sets `group: <workflow>-<arch>-<github.ref>` with
    cancel-in-progress. Two dispatches of the same commit on the same branch
    therefore share a group, and the second kills the first. Different branch,
    different group, and the two runs proceed in parallel.
    """
    suffix = "" if maxschedchunk is None else f"-c{maxschedchunk}"
    if arch and arch != "blackhole":
        suffix = f"-{arch[:2]}{suffix}"
    if apply_ref:
        suffix = f"-{short(git('rev-parse', apply_ref))[:7]}{suffix}"
    if runner_opts:
        # Name the variant, or a single-core run overwrites the 15-worker one.
        for tag, key in (
            ("n", "workers"),
            ("s", "split_into"),
            ("g", "slice_group"),
            ("rc", "run_count"),
        ):
            if runner_opts.get(key) is not None:
                suffix += f"-{tag}{runner_opts[key]}"
    branch = f"{BRANCH_PREFIX}{short(sha)}{suffix}-r{index}"
    head, _ = force_non_sol(sha, maxschedchunk, apply_ref, runner_opts)
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


def start_runs(sha, count, args_ns=None):
    """Dispatch one run per branch and return their ids."""
    inputs = dispatch_inputs(sha, getattr(args_ns, "arch", None) or "blackhole")
    print(f"  dispatch inputs: {inputs}")
    ids = []
    for i in range(1, count + 1):
        branch = push_branch(
            sha,
            i,
            getattr(args_ns, "maxschedchunk", None),
            getattr(args_ns, "apply", None),
            runner_opts_of(args_ns) if args_ns else None,
            getattr(args_ns, "arch", None) if args_ns else None,
        )
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
    key = variant_key(sha, args)
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
        run_ids = start_runs(sha, args.runs, args)
    wait_for(run_ids)

    bad = [r for r in run_ids if run_conclusion(r) != "success"]
    if bad:
        raise RuntimeError(
            f"run(s) {bad} did not succeed; a measurement needs two clean runs"
        )

    work = pathlib.Path(args.work_dir) / key.replace("+", "_")
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
        dispatch_inputs=dispatch_inputs(
            sha, getattr(args, "arch", None) or "blackhole"
        ),
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
    ap.add_argument(
        "--workers",
        type=int,
        help="set -n on both perf passes. 1 gives a single xdist worker, so one "
        "Tensix with nothing running concurrently beside it — which separates "
        "cross-core interference from state carried between tests on one core",
    )
    ap.add_argument("--test-filter", help="pytest -k expression for both passes")
    ap.add_argument(
        "--split-into",
        type=int,
        help="replace the shard split; group 1 runs that slice and every other "
        "group exits, so one card runs one sequence",
    )
    ap.add_argument(
        "--run-count",
        type=int,
        help="measure each point this many times in one run and record min() beside mean(). Interference from neighbouring cores can only add cycles, so min is the aggregation that should survive it",
    )
    ap.add_argument(
        "--arch",
        default="blackhole",
        choices=("blackhole", "wormhole"),
        help="which architecture to measure. Everything so far is Blackhole; #53763 measured Wormhole as noisier on a single card, so whether the chunk effect is arch-specific is an open question",
    )
    ap.add_argument(
        "--slice-group",
        type=int,
        help="which group of --split-into to measure (default 1). Group 2 of 20 "
        "straddles the matmul boundary, so it mixes modules — which a "
        "single-module slice cannot, and heterogeneity is what is under test",
    )
    ap.add_argument(
        "--apply",
        help="also apply this commit's changes (e.g. a fix PR's head) on top of "
        "the commit being measured, so a candidate fix is measured by exactly "
        "the same procedure as everything else",
    )
    ap.add_argument(
        "--maxschedchunk",
        type=int,
        help="also patch pytest.ini to this xdist scheduling chunk. #53642 raised "
        "it from 10 to 2000, which rewrote every test's predecessors on its core; "
        "restoring 10 tests whether the instability is scheduling order or the "
        "per-worker core mapping that landed in the same commit",
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
