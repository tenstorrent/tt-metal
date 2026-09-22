#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SCRATCH -- untracked on purpose; do not commit, do not add to a PR.
#
"""Bisect the user count a prefill pipeline can actually serve, one full relaunch per candidate.

``dflash_forward_pressure`` run in-process can only estimate: the model is built once, at one user
count, and ``build_runtime`` bakes that count into the model config as ``slot_num``. Reallocating
bigger caches around an already-built model would measure a configuration that cannot be served.
So each candidate gets its own launch, where every layer of the stack is sized at N.

A launch is classified from the per-rank ``SERVED`` lines:

  pass    every rank reported ok=1
  fail    some rank reported ok=0 -- N does not fit or does not run
  error   no verdict at all -- the run died before it could decide, which is not evidence about N

An error is retried once behind a device reset rather than folded into the bisection, because
treating infrastructure noise as a capacity limit silently reports a number that is too low.

State is checkpointed after every launch, so an interrupted sweep resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path("/data/nmilicevic/tt-metal")
HOSTS = ["bh-glx-120-b06u02", "bh-glx-120-b06u08", "bh-glx-120-b07u02", "bh-glx-120-b07u08"]
LAUNCH_TIMEOUT_S = 45 * 60
SERVED_RE = re.compile(r"SERVED rank=(\d+) num_ranks=(\d+) users=(\d+) fit=(\d) ok=(\d)")
LOGDIR_RE = re.compile(r"^logs: (\S+)", re.M)


def reset_quad() -> None:
    """Reset every host at once; a staggered reset leaves the cross-host eth links out of sync."""
    procs = [
        subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", h, "tt-smi -glx_reset_auto"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for h in HOSTS
    ]
    for p in procs:
        p.wait()


def launch(users: int, arm: str, positions: int, mode: str = "confirm") -> tuple[str, str, Path | None]:
    scored_on_fit = mode == "fit"
    env_prefix = (
        f"MODULE=dflash_forward_pressure BASE_USERS={users} "
        f"PREFILL_PRESSURE_MODE={mode} PREFILL_PRESSURE_POSITIONS={positions}"
    )
    cmd = f"source python_env/bin/activate && {env_prefix} ./run_dflash_capacity.sh {arm}"
    try:
        res = subprocess.run(
            ["bash", "-lc", cmd], cwd=REPO, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT_S
        )
        out = res.stdout + res.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode(errors="replace") + (exc.stderr or b"").decode(errors="replace")
        return "error", "launch timed out", _logdir(out)

    logdir = _logdir(out)
    verdicts = {}
    num_ranks = 0
    for path in sorted((logdir / "ranklogs").glob("*")) if logdir else []:
        for rank, ranks, got_users, fit, ok in SERVED_RE.findall(path.read_text(errors="replace")):
            if int(got_users) != users:
                return "error", f"rank {rank} reported users={got_users}, expected {users}", logdir
            num_ranks = int(ranks)
            # `fit` mode stops after the caches, so scoring on `ok` there would call every launch a
            # failure. The scored field is whichever stage the mode actually ran.
            verdicts[int(rank)] = int(fit) if scored_on_fit else int(ok)

    if not verdicts or len(verdicts) < num_ranks:
        return "error", f"{len(verdicts)}/{num_ranks or '?'} ranks reported a verdict", logdir
    if all(verdicts.values()):
        return "pass", f"{len(verdicts)} ranks ok", logdir
    failed = sorted(r for r, ok in verdicts.items() if not ok)
    return "fail", f"ranks {failed} could not serve {users}", logdir


def _logdir(out: str) -> Path | None:
    m = LOGDIR_RE.search(out)
    return Path(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dflash", choices=["dflash", "plain"])
    ap.add_argument("--lo", type=int, default=1, help="highest count already known to serve")
    ap.add_argument("--hi", type=int, default=96, help="lowest count already known to fail")
    ap.add_argument("--seed", type=int, help="first candidate; defaults to the midpoint")
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--mode", default="confirm", choices=["confirm", "fit"])
    ap.add_argument("--state", default="/data/nmilicevic/dflash_capacity/bisect_state.json")
    args = ap.parse_args()

    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("arm") == args.arm:
        lo, hi = state["lo"], state["hi"]
        trials = state["trials"]
        print(f"resuming {args.arm}: lo={lo} hi={hi} after {len(trials)} launches")
    else:
        lo, hi, trials = args.lo, args.hi, []

    candidate = args.seed if args.seed and not trials else None
    while hi - lo > 1:
        n = candidate if candidate is not None else (lo + hi) // 2
        candidate = None
        n = min(max(n, lo + 1), hi - 1)

        t0 = time.time()
        verdict, detail, logdir = launch(n, args.arm, args.positions, args.mode)
        mins = (time.time() - t0) / 60
        print(f"[{time.strftime('%H:%M:%S')}] users={n} -> {verdict} ({detail}) in {mins:.1f} min  {logdir}")

        if verdict == "error":
            if any(t["users"] == n and t["verdict"] == "error" for t in trials):
                print(f"users={n} errored twice; stopping rather than guessing a verdict")
                break
            trials.append({"users": n, "verdict": verdict, "detail": detail, "minutes": round(mins, 1)})
            reset_quad()
            candidate = n
        else:
            trials.append({"users": n, "verdict": verdict, "detail": detail, "minutes": round(mins, 1)})
            if verdict == "pass":
                lo = n
            else:
                hi = n
                reset_quad()

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"arm": args.arm, "lo": lo, "hi": hi, "trials": trials}, indent=2))

    print(f"\nBISECT arm={args.arm} max_served_users={lo} first_failing={hi} launches={len(trials)}")
    for t in trials:
        print(f"  users={t['users']:>3}  {t['verdict']:<5}  {t['minutes']:>5} min  {t['detail']}")


if __name__ == "__main__":
    main()
