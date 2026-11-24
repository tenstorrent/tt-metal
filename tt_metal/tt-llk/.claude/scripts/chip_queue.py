#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Queue LLK test runs across several Blackhole chips on one host.

Every job goes through two stages:
  1. compile  - `run_test.sh compile` on a CPU slot, into a private build tree (RUNNER_TEMP),
                so concurrent jobs from different worktrees never clobber each other's ELFs.
  2. simulate - `run_test.sh simulate` on one chip: TT_VISIBLE_DEVICES=<chip> (the harness then sees
                it as device 0), TT_LLK_CHIP=<chip> (a hang resets only this chip), and a per-chip lock.

Usage (from anywhere):
  chip_queue.py daemon [--chips 0,1,2,3] [--compile-slots 3] [--jobs 5]
  chip_queue.py submit --worktree <tt-llk dir> --test test_x.py [--impl hal|tt|<dir>] [--perf]
                       [--k EXPR] [--test-id ID] [--maxfail N] [--timeout S] [--tag TAG] [--priority P]
                       [--owner NAME] [--note TEXT]
  chip_queue.py wait <job-id> [--timeout S]      # prints the result line, exit 0 only on PASS
  chip_queue.py run  <same args as submit>        # submit + wait
  chip_queue.py status [--json]

--impl hal uses the worktree's tests/hal_llk (TT_LLK_IMPL_INCLUDE_DIRS=hal_llk); tt (default) builds
production; any other value is passed through as the include dir.
"""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("CHIP_QUEUE_ROOT", str(Path.home() / "bh-port" / "queue")))
STAGES = ["pending", "compiling", "compiled", "running", "done"]
LOCK = threading.Lock()


def _dirs():
    for s in STAGES + ["logs", "runner"]:
        (ROOT / s).mkdir(parents=True, exist_ok=True)


def _write(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _find(job_id):
    for s in STAGES:
        p = ROOT / s / f"{job_id}.json"
        if p.exists():
            return s, p
    return None, None


def _move(job, src, dst):
    with LOCK:
        job["stage"] = dst
        job[f"t_{dst}"] = time.time()
        _write(ROOT / dst / f"{job['id']}.json", job)
        (ROOT / src / f"{job['id']}.json").unlink(missing_ok=True)


def _claim(stage, dst, **fields):
    """Atomically take the highest-priority, oldest job in `stage` and move it to `dst`.
    The move happens under the same lock as the pick, so two workers can never take the same job.
    """
    with LOCK:
        jobs = [j for j in (_read(p) for p in (ROOT / stage).glob("*.json")) if j]
        if not jobs:
            return None
        jobs.sort(key=lambda j: (-j.get("priority", 0), j["t_submit"]))
        job = jobs[0]
        job.update(fields, stage=dst)
        job[f"t_{dst}"] = time.time()
        _write(ROOT / dst / f"{job['id']}.json", job)
        (ROOT / stage / f"{job['id']}.json").unlink(missing_ok=True)
        return job


def _base_cmd(job, command):
    script = Path(job["worktree"]) / ".claude/scripts/run_test.sh"
    cmd = [
        "bash",
        str(script),
        command,
        "--worktree",
        job["worktree"],
        "--arch",
        "blackhole",
        "--test",
        job["test"],
        "--timeout",
        str(job.get("timeout", 1800)),
    ]
    if job.get("k"):
        cmd += ["--k", job["k"]]
    if job.get("test_id"):
        cmd += ["--test-id", job["test_id"]]
    if job.get("perf"):
        cmd += ["--perf", "--run-tag", job["tag"], "--keep-runs", "500"]
    return cmd


def _env(job):
    env = dict(os.environ)
    env["RUNNER_TEMP"] = str(ROOT / "runner" / job["id"])
    impl = job.get("impl", "tt")
    env.pop("TT_LLK_IMPL_INCLUDE_DIRS", None)
    if impl == "hal":
        env["TT_LLK_IMPL_INCLUDE_DIRS"] = "hal_llk"
    elif impl != "tt":
        env["TT_LLK_IMPL_INCLUDE_DIRS"] = impl
    return env


def _summarize(log_text):
    verdict = re.findall(r"RUN_LLK_TESTS_VERDICT === (\w+)", log_text)
    counts = re.findall(
        r"=+ (.*?(?:passed|failed|error|skipped|deselected).*?) in [0-9.]+s", log_text
    )
    report = re.findall(r"RUN_LLK_TESTS_PERF_REPORT === (\S+)", log_text)
    errors = re.findall(r"error: .{0,200}", log_text)
    return {
        "verdict": verdict[-1] if verdict else "UNKNOWN",
        "counts": counts[-1] if counts else "",
        "perf_report": report[-1] if report else "",
        "first_error": errors[0] if errors else "",
    }


def _compile_worker(slot, jobs_per_compile):
    while True:
        job = _claim("pending", "compiling", compile_slot=slot)
        if not job:
            time.sleep(2)
            continue
        log = ROOT / "logs" / f"{job['id']}.compile.log"
        cmd = _base_cmd(job, "compile") + ["--jobs", str(jobs_per_compile)]
        with open(log, "w") as f:
            rc = subprocess.call(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=_env(job),
                cwd=job["worktree"],
            )
        res = _summarize(log.read_text(errors="replace"))
        job["compile_log"] = str(log)
        if rc != 0 or res["verdict"] not in ("PASS", "UNKNOWN"):
            job.update(
                res,
                rc=rc,
                verdict=(
                    "COMPILE_FAIL"
                    if res["verdict"] in ("UNKNOWN", "PASS")
                    else res["verdict"]
                ),
            )
            _move(job, "compiling", "done")
        else:
            _move(job, "compiling", "compiled")


def _chip_worker(chip):
    while True:
        job = _claim("compiled", "running", chip=chip)
        if not job:
            time.sleep(1)
            continue
        log = ROOT / "logs" / f"{job['id']}.run.log"
        env = _env(job)
        env["TT_VISIBLE_DEVICES"] = str(chip)
        env["TT_LLK_CHIP"] = str(chip)
        cmd = _base_cmd(job, "simulate") + ["--lock", f"/tmp/tt-llk-chip{chip}.lock"]
        if job.get("maxfail"):
            cmd += ["--maxfail", str(job["maxfail"])]
        with open(log, "w") as f:
            rc = subprocess.call(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=job["worktree"]
            )
        job.update(_summarize(log.read_text(errors="replace")), rc=rc, run_log=str(log))
        _move(job, "running", "done")


def _status_obj():
    out = {"t": time.time(), "stages": {}, "running": [], "recent": []}
    for s in STAGES:
        js = [j for j in (_read(p) for p in (ROOT / s).glob("*.json")) if j]
        out["stages"][s] = len(js)
        if s == "running":
            out["running"] = [
                {
                    k: j.get(k)
                    for k in (
                        "id",
                        "chip",
                        "owner",
                        "test",
                        "impl",
                        "perf",
                        "t_running",
                    )
                }
                for j in js
            ]
        if s == "done":
            js.sort(key=lambda j: j.get("t_done", 0), reverse=True)
            out["recent"] = [
                {
                    k: j.get(k)
                    for k in (
                        "id",
                        "owner",
                        "test",
                        "impl",
                        "perf",
                        "verdict",
                        "counts",
                        "chip",
                    )
                }
                for j in js[:30]
            ]
    return out


def _status_writer():
    while True:
        try:
            _write(ROOT / "status.json", _status_obj())
        except Exception:  # noqa: BLE001 - keep the daemon alive
            pass
        time.sleep(15)


def cmd_daemon(a):
    _dirs()
    # One daemon per queue root: a second one would hand the same chip to two jobs.
    lock = open(ROOT / "daemon.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another chip_queue daemon already serves {ROOT}")
    lock.write(str(os.getpid()))
    lock.flush()
    # Jobs left mid-flight by a previous daemon go back to the start of their stage.
    for s, back in (("compiling", "pending"), ("running", "compiled")):
        for p in (ROOT / s).glob("*.json"):
            j = _read(p)
            if j:
                _move(j, s, back)
    chips = [int(c) for c in a.chips.split(",")]
    threads = [
        threading.Thread(target=_compile_worker, args=(i, a.jobs), daemon=True)
        for i in range(a.compile_slots)
    ]
    threads += [
        threading.Thread(target=_chip_worker, args=(c,), daemon=True) for c in chips
    ]
    threads.append(threading.Thread(target=_status_writer, daemon=True))
    for t in threads:
        t.start()
    print(
        f"chip_queue daemon: chips {chips}, {a.compile_slots} compile slots, root {ROOT}",
        flush=True,
    )
    while True:
        time.sleep(60)


def _new_job(a):
    _dirs()
    worktree = str(Path(a.worktree).resolve())
    if not (Path(worktree) / ".claude/scripts/run_test.sh").exists():
        sys.exit(
            f"--worktree must be a tt-llk directory (no .claude/scripts/run_test.sh under {worktree})"
        )
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = {
        "id": jid,
        "worktree": worktree,
        "test": a.test,
        "impl": a.impl,
        "perf": a.perf,
        "k": a.k,
        "test_id": a.test_id,
        "maxfail": a.maxfail,
        "timeout": a.timeout,
        "priority": a.priority,
        "owner": a.owner,
        "note": a.note,
        "tag": a.tag or f"q-{jid}",
        "t_submit": time.time(),
        "stage": "pending",
    }
    cached = _cached_tt(job) if not getattr(a, "no_cache", False) else None
    if cached:
        job.update(
            {
                k: cached.get(k)
                for k in ("verdict", "counts", "run_log", "compile_log", "chip")
            },
            stage="done",
            cached_from=cached["id"],
            t_done=time.time(),
        )
        _write(ROOT / "done" / f"{jid}.json", job)
        return jid
    _write(ROOT / "pending" / f"{jid}.json", job)
    return jid


def _cached_tt(job):
    """A passed functional run of the same production (tt) test. Work-package branches never change production
    headers, so a tt result is the same whichever worktree it ran from; reuse it instead of using a chip again.
    """
    if job["impl"] != "tt" or job["perf"]:
        return None
    key = ("test", "k", "test_id")  # maxfail only matters when something fails
    for p in sorted((ROOT / "done").glob("*.json"), reverse=True):
        try:
            d = _read(p)
        except Exception:
            continue
        if (
            d.get("impl") == "tt"
            and not d.get("perf")
            and d.get("verdict") == "PASS"
            and all(d.get(k) == job[k] for k in key)
        ):
            return d
    return None


def _line(j):
    return (
        f"{j['id']} {j.get('verdict', '?')} chip={j.get('chip', '-')} {j.get('test')} impl={j.get('impl')} "
        f"{'perf ' if j.get('perf') else ''}{j.get('counts', '')} report={j.get('perf_report', '')} "
        f"log={j.get('run_log') or j.get('compile_log', '')} {j.get('first_error', '')}"
    )


def cmd_wait(job_id, timeout):
    t0 = time.time()
    while True:
        stage, p = _find(job_id)
        if stage is None:
            sys.exit(f"unknown job {job_id}")
        if stage == "done":
            j = _read(p)
            print(_line(j))
            return 0 if j.get("verdict") == "PASS" else 1
        if timeout and time.time() - t0 > timeout:
            print(f"{job_id} still {stage} after {timeout}s")
            return 2
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("daemon")
    d.add_argument("--chips", default="0,1,2,3")
    d.add_argument("--compile-slots", type=int, default=3)
    d.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="pytest compile parallelism per compile slot",
    )
    for name in ("submit", "run"):
        s = sp.add_parser(name)
        s.add_argument("--worktree", required=True)
        s.add_argument("--test", required=True)
        s.add_argument("--impl", default="tt")
        s.add_argument("--perf", action="store_true")
        s.add_argument("--k")
        s.add_argument("--test-id")
        s.add_argument("--maxfail", type=int)
        s.add_argument("--timeout", type=int, default=1800)
        s.add_argument("--tag")
        s.add_argument("--priority", type=int, default=0)
        s.add_argument(
            "--no-cache",
            action="store_true",
            help="run a tt job even if an identical one already passed",
        )
        s.add_argument("--owner", default=os.environ.get("USER", "?"))
        s.add_argument("--note", default="")
        s.add_argument("--wait-timeout", type=int, default=0)
    w = sp.add_parser("wait")
    w.add_argument("job_id")
    w.add_argument("--timeout", type=int, default=0)
    st = sp.add_parser("status")
    st.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.cmd == "daemon":
        cmd_daemon(a)
    elif a.cmd == "submit":
        print(_new_job(a))
    elif a.cmd == "run":
        jid = _new_job(a)
        print(f"submitted {jid}", flush=True)
        sys.exit(cmd_wait(jid, a.wait_timeout))
    elif a.cmd == "wait":
        sys.exit(cmd_wait(a.job_id, a.timeout))
    elif a.cmd == "status":
        _dirs()
        s = _status_obj()
        if a.json:
            print(json.dumps(s, indent=1))
        else:
            print("stages:", " ".join(f"{k}={v}" for k, v in s["stages"].items()))
            for r in s["running"]:
                print(
                    f"  chip {r['chip']}: {r['id']} {r['owner']} {r['test']} impl={r['impl']}{' perf' if r['perf'] else ''}"
                )
            for r in s["recent"][:10]:
                print(
                    f"  done {r['id']} {r['verdict']} chip={r['chip']} {r['owner']} {r['test']} impl={r['impl']} {r['counts']}"
                )


if __name__ == "__main__":
    main()
