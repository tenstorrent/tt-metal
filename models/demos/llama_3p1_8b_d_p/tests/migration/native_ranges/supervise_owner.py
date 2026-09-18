"""Node task entry point. Never wrap this process group in a forced timeout.

Normal/cancel phases are bounded. An ambiguous native stop enters a recorded hold,
keeping the Slurm task, owner and lock alive until root operates recovery.
"""
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

from native_observation import process_identity
from peer_stop import read_peer_event
from runner_support import require, sha256, validate_plan, write_json
from supervision import root_release_valid, supervise

ACTIVE = None


def main():
    global ACTIVE
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--role", choices=("source", "passive"), required=True)
    args = parser.parse_args()
    require(sha256(args.plan) == args.plan_sha256, "Plan bytes changed")
    plan = json.loads(args.plan.read_bytes())
    validate_plan(plan)
    local, role = plan[args.role], args.role
    require(socket.gethostname() == local["host"], "Wrong assigned node")
    require(os.environ.get("SLURM_JOB_ID") == str(local["job_id"]), "Wrong assigned job")
    budget = plan["supervisor"]
    lease = dt.datetime.fromisoformat(local["lease_end_utc"])
    remaining = (lease - dt.datetime.now(dt.timezone.utc)).total_seconds()
    require(
        remaining >= sum(budget[k] for k in ("normal_seconds", "cancel_seconds", "recovery_reserve_seconds")),
        "Lease does not cover normal phases and root recovery reserve",
    )
    lock = Path(local["node_lock"]).open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Only this node task owns the shared flock. The owner inherits the same open
    # file description, so losing the supervisor cannot silently release its lock.
    output = Path(plan["run_dir"]) / (role + "-supervisor")
    output.mkdir(exist_ok=False)
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    env = dict(os.environ, MIGRATION_NODE_LOCK_FD=str(lock.fileno()))
    log = (output / "owner.log").open("xb")
    argv = [
        sys.executable,
        str(Path(__file__).with_name("owner_runner.py")),
        "--plan",
        str(args.plan),
        "--plan-sha256",
        args.plan_sha256,
        "--role",
        role,
    ]
    owner = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(lock.fileno(),))
    ACTIVE = dict(owner=owner, lock=lock, log=log, release_allowed=False, output=str(output))
    try:
        identity = process_identity(owner.pid)
    except OSError:
        identity = dict(pid=owner.pid, start_ticks=None, observation="owner vanished before identity read")
    started = dict(
        run_nonce=plan["run_nonce"],
        role=role,
        supervisor=process_identity(os.getpid()),
        owner=identity,
        argv=argv,
        node_lock=local["node_lock"],
        lease_end_utc=local["lease_end_utc"],
        budgets=budget,
    )
    endpoint = Path(plan["run_dir"]) / role

    def state():
        result = {}
        for key, name in [
            ("result", "result.json"),
            ("recovery", "recovery-required.json"),
            ("failure", "failure.json"),
        ]:
            path = endpoint / name
            if path.exists():
                value = json.loads(path.read_bytes())
                require(value.get("run_nonce") == plan["run_nonce"], "Foreign/stale owner receipt")
                if key == "result":
                    recorded = value.get("owner", {})
                    require(
                        (recorded.get("pid"), recorded.get("start_ticks"))
                        == (identity["pid"], identity["start_ticks"]),
                        "Final receipt belongs to another owner",
                    )
                result[key] = value
        peer = read_peer_event(Path(plan["run_dir"]), role, plan["run_nonce"])
        if peer is not None:
            result[peer["kind"]] = peer
        return result

    def request_stop(reason):
        path = Path(plan["run_dir"]) / (role + "-stop-requested.json")
        if not path.exists():
            write_json(path, dict(run_nonce=plan["run_nonce"], role=role, reason=reason))

    try:
        write_json(output / "started.json", started)
        decision = supervise(
            owner,
            state,
            request_stop,
            normal_seconds=budget["normal_seconds"],
            cancel_seconds=budget["cancel_seconds"],
            lease_seconds=remaining,
            recovery_reserve=budget["recovery_reserve_seconds"],
            external_stop=lambda: stop["requested"],
        )
    except BaseException as error:
        request_stop("supervisor_observation_failure")
        decision = dict(
            phase="recovery_hold", reason=repr(error), release_lock=False, buffers_retained=owner.poll() is None
        )
    decision.update(
        run_nonce=plan["run_nonce"],
        role=role,
        owner=identity,
        endpoints={r: plan[r] for r in ("source", "passive")},
        lease_end_utc=local["lease_end_utc"],
    )
    write_json(output / "decision.json", decision)
    if decision["release_lock"]:
        ACTIVE["release_allowed"] = True
        lock.close()
        log.close()
        return decision["owner_exit"]
    print("ROOT_RECOVERY_REQUIRED: owner/node lock retained; see " + str(output / "decision.json"), flush=True)
    # Returning the Slurm main task could kill its remaining owner. Keep this rank
    # alive; root must recover both endpoints before terminating either held step.
    # Slurm expiry/SIGKILL can still kill buffers; this is not survival past lease.
    while True:
        if owner.poll() is not None:
            try:
                final = state().get("result", {})
                if final.get("owner_cleanup_complete") is True and not final.get("cleanup_errors"):
                    write_json(
                        output / "hold-resolved-clean.json",
                        dict(run_nonce=plan["run_nonce"], owner=identity, exit=owner.returncode, result=final),
                    )
                    break
            except (OSError, ValueError, RuntimeError):
                pass
        release = output / "root-release.json"
        if release.exists() and owner.poll() is not None:
            try:
                receipt = json.loads(release.read_bytes())
                if root_release_valid(receipt, identity, plan["run_nonce"], owner.returncode, local):
                    break
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        time.sleep(1)
    ACTIVE["release_allowed"] = True
    lock.close()
    log.close()
    return 75


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        traceback.print_exc()
        if ACTIVE is None or ACTIVE["release_allowed"]:
            raise
        # Even a receipt-write/parser failure after launch must not unwind the
        # Slurm main task and cause cleanup to kill an ambiguous native owner.
        print(
            "ROOT_RECOVERY_REQUIRED supervisor error; owner PID="
            + str(ACTIVE["owner"].pid)
            + " output="
            + ACTIVE["output"]
            + "; node lock retained",
            flush=True,
        )
        while True:
            time.sleep(1)
    raise SystemExit(status)
