"""Reuse the accepted node-owned flock/owner hold protocol; no forced timeout or kill."""
import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from process_identity import process_identity
from runtime_launch import bind_owner_plan, owner_state, validate
from supervision import root_release_valid, supervise
from support import require, sha256, write_json

ACTIVE = None


def main():
    global ACTIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    a = ap.parse_args()
    require(sha256(a.plan) == a.plan_sha256, "root plan changed")
    plan = json.loads(a.plan.read_bytes())
    validate(plan)
    require(socket.gethostname() == plan["node"] and os.environ["SLURM_JOB_ID"] == plan["job_id"], "wrong node/job")
    require(len(os.sched_getaffinity(0)) == 1, "one inherited CPU required")
    scheduler = json.loads(Path(plan["scheduler_receipt"]).read_bytes())
    require(sha256(plan["scheduler_receipt"]) == plan["scheduler_receipt_sha256"], "scheduler receipt changed")
    require(
        all(scheduler.get(k) == plan[k] for k in ("node", "job_id", "run_nonce", "lease_end_utc")),
        "scheduler binding changed",
    )
    now = dt.datetime.now(dt.timezone.utc)
    end = dt.datetime.fromisoformat(plan["lease_end_utc"])
    require(
        0 <= (now - dt.datetime.fromisoformat(scheduler["checked_utc"])).total_seconds() <= 120, "scheduler check stale"
    )
    require(
        scheduler["owner"] == plan["allocation_owner"]
        and scheduler["state"] == "RUNNING"
        and scheduler["steps_empty"] is True,
        "scheduler ownership missing",
    )
    require((end - now).total_seconds() >= 2700, "insufficient owner and recovery lease")
    lock = Path(plan["lock_path"]).open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = Path(plan["supervisor_dir"])
    output.mkdir(exist_ok=False)
    here = Path(__file__).parent
    pins_path = output / "bound-source-pins.json"
    write_json(pins_path, plan["source_pins"])
    env = dict(
        os.environ,
        WRITER_EXPECTED_NODE=plan["node"],
        WRITER_EXPECTED_JOB=plan["job_id"],
        WRITER_SOURCE_PINS=str(pins_path),
    )
    # This accepted stdlib guard scans other workloads/FDs before any owner/native import.
    subprocess.run(
        ["python3", "-I", "-S", "-B", str(here / "node-preflight.py"), str(output), "device"],
        env=env,
        check=True,
        timeout=120,
    )
    now = dt.datetime.now(dt.timezone.utc)
    receipt = dict(
        node=plan["node"],
        job_id=plan["job_id"],
        run_nonce=plan["run_nonce"],
        checked_utc=now.isoformat(),
        lease_end_utc=plan["lease_end_utc"],
        scheduler_owner_verified=True,
        empty_steps_verified=True,
        no_competing_device_fds=True,
        lock_held=True,
        root_plan_sha256=a.plan_sha256,
        scheduler_receipt_sha256=plan["scheduler_receipt_sha256"],
    )
    receipt_path = output / "launch-receipt.json"
    write_json(receipt_path, receipt)
    owner_plan = bind_owner_plan(plan, receipt, receipt_path)
    owner_path = output / "owner-plan.json"
    write_json(owner_path, owner_plan)
    owner_hash = sha256(owner_path)
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    log_path = output / "owner.log"
    log = log_path.open("xb")
    env["EDGE_NODE_LOCK_FD"] = str(lock.fileno())
    argv = [sys.executable, "-B", str(here / "run_owner.py"), "--plan", str(owner_path), "--plan-sha256", owner_hash]
    owner = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
    ACTIVE = dict(owner=owner, lock=lock, log=log, release_allowed=False, output=str(output))
    identity = process_identity(owner.pid)
    write_json(
        output / "started.json",
        dict(
            owner=identity,
            supervisor=process_identity(os.getpid()),
            argv=argv,
            owner_plan_sha256=owner_hash,
            run_nonce=plan["run_nonce"],
            lease_end_utc=plan["lease_end_utc"],
        ),
    )

    def state():
        return owner_state(Path(plan["output"]) / "report.json", log_path, plan, owner_hash)

    def request_stop(reason):
        path = output / "stop-requested.json"
        if not path.exists():
            write_json(path, dict(run_nonce=plan["run_nonce"], reason=reason, owner=identity))
        # The frozen owner has no cooperative cancellation hook. Do not signal it:
        # allow its existing finally to finish, then retain on an ambiguous deadline.

    remaining = (end - dt.datetime.now(dt.timezone.utc)).total_seconds()
    decision = supervise(
        owner,
        state,
        request_stop,
        normal_seconds=1800,
        cancel_seconds=300,
        lease_seconds=remaining,
        recovery_reserve=600,
        external_stop=lambda: stop["requested"],
    )
    decision.update(
        owner=identity,
        run_nonce=plan["run_nonce"],
        node=plan["node"],
        job_id=plan["job_id"],
        lease_end_utc=plan["lease_end_utc"],
    )
    write_json(output / "decision.json", decision)
    if decision["release_lock"]:
        ACTIVE["release_allowed"] = True
        log.close()
        lock.close()
        return decision["owner_exit"]
    print("ROOT_RECOVERY_REQUIRED: owner and node lock retained; " + str(output / "decision.json"), flush=True)
    endpoint = dict(job_id=plan["job_id"], host=plan["node"], node_lock=plan["lock_path"])
    while True:
        if owner.poll() is not None:
            try:
                final = state().get("result", {})
                if final.get("owner_cleanup_complete") is True:
                    write_json(
                        output / "hold-resolved-clean.json",
                        dict(owner=identity, owner_exit=owner.returncode, result=final),
                    )
                    break
                release = output / "root-release.json"
                if release.exists() and root_release_valid(
                    json.loads(release.read_bytes()), identity, plan["run_nonce"], owner.returncode, endpoint
                ):
                    break
            except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
                pass
        time.sleep(1)
    ACTIVE["release_allowed"] = True
    log.close()
    lock.close()
    return 75


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        traceback.print_exc()
        if ACTIVE is None or ACTIVE["release_allowed"]:
            raise
        print("ROOT_RECOVERY_REQUIRED: supervisor exception; owner/lock retained " + str(ACTIVE["output"]), flush=True)
        while True:
            time.sleep(1)
    raise SystemExit(status)
