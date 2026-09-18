"""Two assigned nodes, one retained owner each; no forced timeout or automatic reset."""

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from runner_support import require, sha256, validate_plan, write_json


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=30, env=dict(os.environ, TZ="UTC"))
    require(result.returncode == 0, repr((argv, result.returncode, result.stderr)))
    return result.stdout


def validate_shared_assignment(plan, role, assignment):
    local = plan[role]
    require(
        assignment == plan["allocation_owner"] + "|RUNNING|" + plan["allocation_nodelist"],
        "Wrong current shared assignment: " + role,
    )
    require(
        local["host"] in plan["allocation_nodes"] and len(plan["allocation_nodes"]) == 2,
        "Selected endpoint is outside exact two-node allocation: " + role,
    )


def preflight(plan, role):
    local = plan[role]
    job, host = str(local["job_id"]), local["host"]
    assignment = command(["squeue", "-h", "-j", job, "-o", "%u|%T|%N"]).strip()
    validate_shared_assignment(plan, role, assignment)
    steps = command(["squeue", "--steps", "-h", "-j", job, "-o", "%i|%j|%N"])
    require(not steps.strip(), "Assigned node has active steps: " + role)
    raw = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(x.split("=", 1) for x in raw.split() if "=" in x)
    end = dt.datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=dt.timezone.utc)
    require(end == dt.datetime.fromisoformat(local["lease_end_utc"]), "Frozen lease differs from scheduler")
    required = (
        sum(plan["supervisor"][k] for k in ("normal_seconds", "cancel_seconds", "recovery_reserve_seconds")) + 120
    )
    require((end - dt.datetime.now(dt.timezone.utc)).total_seconds() >= required, "Insufficient lease/recovery reserve")
    health = json.loads(Path(local["health_receipt"]).read_bytes())
    require(
        health["actual_exit"] == health["verified_exit"] == 0
        and health["clean_close"]
        and str(health["job_id"]) == job
        and health["node"] == host,
        "Endpoint health not accepted",
    )
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(health["ended_utc"])).total_seconds()
    require(0 <= age < 1500, "Fresh endpoint health/activity required")
    return dict(job=job, node=host, raw=raw, steps=steps, health_age=age, required_seconds=required)


def request_stop(run, plan, role, reason):
    path = run / (role + "-stop-requested.json")
    if not path.exists():
        write_json(path, dict(run_nonce=plan["run_nonce"], role=role, reason=reason))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha256(args.plan) == args.plan_sha256, "Plan changed")
    plan = json.loads(args.plan.read_bytes())
    plan["plan_path"] = str(args.plan)
    validate_plan(plan)
    checked = {role: preflight(plan, role) for role in ("source", "passive")}
    run = Path(plan["run_dir"])
    run.mkdir(exist_ok=False)
    write_json(run / "scheduler-before.json", checked)
    write_json(run / "source-before.json", {p: sha256(p) for p in plan["pins"]})
    processes = {}
    logs = {}
    try:
        for role in ("source", "passive"):
            local = plan[role]
            argv = [
                "srun",
                "--overlap",
                "--exact",
                "--jobid=" + str(local["job_id"]),
                "--nodes=1",
                "--ntasks=1",
                "--nodelist=" + local["host"],
                "--cpus-per-task=1",
                "bash",
                str(Path(__file__).with_name("node-run.sh")),
                str(args.plan),
                args.plan_sha256,
                role,
            ]
            logs[role] = (run / (role + "-dispatch.log")).open("xb")
            processes[role] = subprocess.Popen(
                argv, stdout=logs[role], stderr=subprocess.STDOUT, start_new_session=True
            )
            write_json(
                run / (role + "-dispatch-started.json"),
                dict(pid=processes[role].pid, argv=argv, started_utc=dt.datetime.now(dt.timezone.utc).isoformat()),
            )
        while any(p.poll() is None for p in processes.values()):
            for role, process in processes.items():
                if process.poll() not in (None, 0):
                    for other in processes:
                        request_stop(run, plan, other, "peer_dispatch_failed")
                path = run / (role + "-supervisor") / "decision.json"
                if path.exists() and json.loads(path.read_bytes()).get("phase") == "recovery_hold":
                    for other in processes:
                        request_stop(run, plan, other, "peer_recovery_hold")
                    write_json(
                        run / "root-action-required.json",
                        dict(
                            run_nonce=plan["run_nonce"],
                            reason="Both owner/step locks deliberately retained as required by native-stop proof",
                            processes={k: v.pid for k, v in processes.items()},
                            endpoints={k: plan[k] for k in processes},
                        ),
                    )
                    print("ROOT_RECOVERY_REQUIRED: no process kill, reset or Slurm cancellation sent", flush=True)
                    return 75
            time.sleep(1)
    except BaseException:
        for role in processes:
            request_stop(run, plan, role, "controller_observation_failure")
        # Detached srun tasks keep their supervisors/owners. Root resumes recovery from saved PIDs.
        raise
    finally:
        for log in logs.values():
            log.close()
    after = {p: sha256(p) for p in plan["pins"]}
    write_json(run / "source-after.json", after)
    exits = {role: p.returncode for role, p in processes.items()}
    write_json(
        run / "dispatch-result.json",
        dict(
            dispatch_exits=exits,
            source_unchanged=after == plan["pins"],
            steps_after={
                r: command(["squeue", "--steps", "-h", "-j", str(plan[r]["job_id"]), "-o", "%i|%j|%N"])
                for r in processes
            },
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        ),
    )
    require(all(x == 0 for x in exits.values()) and after == plan["pins"], "Endpoint failed or source changed")
    from verify_final import verify

    write_json(run / "verified-result.json", verify(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
