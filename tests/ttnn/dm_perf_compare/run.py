#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a DM op before and after a change, and the default against each implementation.

    python tests/ttnn/dm_perf_compare/run.py --calls calls/repeat_smoke.json \\
        --before-tree <build of the merge-base> --after-tree <build of the change> --out results.json

Per round, one fresh process per build (before / after) measures time per call with the profiler
off; the order of the two alternates between rounds. The after build's process measures the default
and both forced implementations. A control pair of the after build against itself, on a few calls,
sets the noise band. A device-profiler pass over the after build then adds device time and the
device ops each implementation dispatches. This script only spawns processes and needs no ttnn.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calls as calls_mod  # noqa: E402
import compare  # noqa: E402
import profile_csv  # noqa: E402
from ops import IMPLS  # noqa: E402

WORKER = os.path.join(HERE, "worker.py")
PROCESS_TIMEOUT_S = 1800


def log(message):
    print(message, file=sys.stderr, flush=True)


def tree_env(tree, cache_dir):
    env = dict(os.environ)
    env["TT_METAL_HOME"] = tree
    env["PYTHONPATH"] = os.pathsep.join([f"{tree}/ttnn", f"{tree}/tools", tree])
    env["TT_METAL_CACHE"] = cache_dir
    return env


def git_sha(tree):
    result = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or None


class Runner:
    def __init__(self, args, doc):
        self.args = args
        self.doc = doc
        self.work = os.path.abspath(args.work_dir)
        os.makedirs(self.work, exist_ok=True)
        self.trees = {"before": os.path.abspath(args.before_tree), "after": os.path.abspath(args.after_tree)}

    def spawn(self, tree_name, label, argv, python_prefix=(), extra_env=None):
        """Run the worker for one build; returns the worker's JSON (or None if the process failed)."""
        out = os.path.join(self.work, f"{label}.json")
        if os.path.exists(out):
            os.remove(out)
        cmd = [self.args.python, *python_prefix, WORKER, *argv, "--calls", self.args.calls, "--out", out]
        env = tree_env(self.trees[tree_name], os.path.join(self.work, f"kernel-cache-{tree_name}"))
        env.update(extra_env or {})
        log("  run %s" % label)
        with open(os.path.join(self.work, f"{label}.log"), "w") as logfile:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.trees[tree_name],
                    env=env,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    timeout=PROCESS_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                log("  %s: timed out" % label)
                return None
        if proc.returncode != 0 or not os.path.exists(out):
            log("  %s: exit %s, see %s.log" % (label, proc.returncode, label))
            return None
        with open(out) as f:
            return json.load(f)

    def time_pass(self, tree_name, label, impls, select, rotation):
        argv = [
            "--pass",
            "time",
            "--impls",
            ",".join(impls),
            "--select",
            ",".join(map(str, select)),
            "--warmup",
            str(self.args.warmup),
            "--reps",
            str(self.args.reps),
            "--rotation",
            str(rotation),
        ]
        return self.spawn(tree_name, label, argv)

    def device_pass(self, label):
        prof = os.path.join(self.work, f"prof-{label}")
        shutil.rmtree(prof, ignore_errors=True)
        argv = ["--pass", "device", "--reps", str(self.args.device_reps)]
        prefix = ("-m", "tracy", "-p", "-r", "-o", prof, "-n", "dmperf")
        # tracy takes the first non-flag token as the program, so the worker path must follow the flags.
        env = {
            "TT_METAL_DEVICE_PROFILER": "1",
            "TRACY_NO_SYS_TRACE": "1",
            "PROFILER_ARTIFACTS_DIR": prof,
            "PROFILER_OUTPUT_DIR": prof + "/reports",
            "PROFILER_LOGS_DIR": prof + "/logs",
        }
        result = self.spawn("after", label, argv, python_prefix=prefix, extra_env=env)
        csv_path = profile_csv.find_csv(prof)
        if result is None or csv_path is None:
            return {}
        return profile_csv.parse_csv(csv_path)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", required=True, help="call-list JSON (see calls.py)")
    parser.add_argument("--before-tree", required=True, help="tt-metal checkout built at the base commit")
    parser.add_argument("--after-tree", required=True, help="tt-metal checkout built with the change")
    parser.add_argument("--out", required=True, help="results JSON")
    parser.add_argument("--work-dir", default="dm_perf_work", help="logs, per-run JSON, kernel caches")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30, help="timed repetitions per call, best kept")
    parser.add_argument("--device-reps", type=int, default=10)
    parser.add_argument("--device-rounds", type=int, default=None, help="default: same as --rounds")
    parser.add_argument("--control-calls", type=int, default=4)
    parser.add_argument("--no-device", action="store_true")
    args = parser.parse_args(argv)
    args.calls = os.path.abspath(args.calls)

    doc = calls_mod.load_calls(args.calls)
    runner = Runner(args, doc)
    count = len(doc["calls"])
    everything = list(range(count))
    control = calls_mod.control_indices(count, args.control_calls)
    control_ids = [doc["calls"][i]["id"] for i in control]
    after_impls = list(IMPLS)

    raw = {
        "time": {"before": [], "after": [], "ctl_before": [], "ctl_after": []},
        "device": [],
        "control_ids": control_ids,
    }
    for rnd in range(args.rounds):
        log("round %d/%d" % (rnd + 1, args.rounds))
        first_is_before = rnd % 2 == 0
        pair = (
            (("before", "before"), ("after", "after"))
            if first_is_before
            else (("after", "after"), ("before", "before"))
        )
        for label, tree in pair:
            impls = ["default"] if label == "before" else after_impls
            raw["time"][label].append(runner.time_pass(tree, f"{label}-r{rnd}", impls, everything, rnd))
        ctl_pair = (("ctl_before", "after"), ("ctl_after", "after"))
        for label, tree in ctl_pair if first_is_before else reversed(ctl_pair):
            raw["time"][label].append(runner.time_pass(tree, f"{label}-r{rnd}", ["default"], control, rnd))

    if not args.no_device:
        for rnd in range(args.device_rounds or args.rounds):
            log("device round %d" % (rnd + 1))
            raw["device"].append(runner.device_pass(f"device-r{rnd}"))

    boards = {r["device_arch"] for runs in raw["time"].values() for r in runs if r and "device_arch" in r}
    meta = {
        "op": doc["op"],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "device_arch": sorted(boards),
        "trees": {name: {"path": path, "sha": git_sha(path)} for name, path in runner.trees.items()},
        "params": {
            "rounds": args.rounds,
            "warmup": args.warmup,
            "reps": args.reps,
            "device_reps": args.device_reps,
            "control_calls": args.control_calls,
        },
    }
    results = compare.build(doc, raw, meta)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    summary = results["summary"]
    log("noise %.2f%%  q1 %s  q2 %s" % (results["noise"]["noise_pct"], summary["q1"], summary["q2"]))
    log("skipped/failed entries: %d" % len(results["skipped"]))


if __name__ == "__main__":
    main(sys.argv[1:])
