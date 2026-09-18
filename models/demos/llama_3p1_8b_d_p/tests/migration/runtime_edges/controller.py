"""Root-reviewed single-node runtime launcher; retain Slurm/owner on ambiguous close."""
import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from edge_guard import check_health
from runtime_launch import validate
from slurm_scope import singleton_srun, target_step_lines, validate_allocation
from support import require, sha256, write_json

HERE = Path(__file__).parent


def text(argv):
    c = subprocess.run(argv, text=True, capture_output=True, timeout=30, env=dict(os.environ, TZ="UTC"))
    require(c.returncode == 0 and not c.stderr.strip(), repr((argv, c.returncode, c.stderr)))
    return c.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    a = ap.parse_args()
    require(os.environ.get("RUNTIME_EDGE_DISPATCH_AUTHORIZED") == "root-reviewed", "root clearance missing")
    require(sha256(a.plan) == a.plan_sha256, "reviewed plan changed")
    plan = json.loads(a.plan.read_bytes())
    validate(plan)
    job, node = plan["job_id"], plan["node"]
    allocation = validate_allocation(
        text(["squeue", "-h", "-j", job, "-o", "%u|%T|%N"]), node, text, expected_owner=plan["allocation_owner"]
    )
    steps = text(["squeue", "--steps", "-h", "-j", job, "-o", "%i|%j|%N"])
    require(not target_step_lines(steps, node, text), "target node busy")
    info = text(["scontrol", "show", "job", job, "-o"])
    fields = dict(x.split("=", 1) for x in info.split() if "=" in x)
    end = dt.datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    require(end == dt.datetime.fromisoformat(plan["lease_end_utc"]), "frozen/live lease differ")
    require((end - now).total_seconds() >= 2820, "2700-second gate plus120-second setup margin required")
    require(sha256(plan["health_receipt"]) == plan["health_receipt_sha256"], "root health bytes changed")
    check_health(json.loads(Path(plan["health_receipt"]).read_bytes()), plan, now)
    before = {k: sha256(k) for k in plan["source_pins"]}
    require(before == plan["source_pins"], "source changed before dispatch")
    run = Path(plan["run_dir"])
    run.mkdir(exist_ok=False)
    write_json(run / "source-before.json", before)
    receipt = dict(
        node=node,
        job_id=job,
        run_nonce=plan["run_nonce"],
        owner=plan["allocation_owner"],
        state="RUNNING",
        steps_empty=True,
        checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        lease_end_utc=end.isoformat(),
        allocation=allocation,
        raw_scontrol=info,
    )
    path = run / "scheduler-before.json"
    write_json(path, receipt)
    # The only added node-plan fields bind fresh scheduler evidence; root-plan bytes stay frozen.
    node_plan = dict(plan, scheduler_receipt=str(path), scheduler_receipt_sha256=sha256(path))
    node_path = run / "node-plan.json"
    write_json(node_path, node_plan)
    node_hash = sha256(node_path)
    argv = singleton_srun(
        job,
        node,
        [
            "python3",
            "-I",
            "-S",
            "-B",
            str(HERE / "single_cpu_exec.py"),
            str(run / "inherited-affinity.json"),
            "bash",
            str(HERE / "node-run.sh"),
            str(node_path),
            node_hash,
        ],
    )
    write_json(run / "command.json", argv)
    with (run / "dispatch.log").open("xb") as log:
        step = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(
            run / "dispatch-started.json",
            dict(pid=step.pid, argv=argv, utc=dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        while step.poll() is None:
            decision = Path(plan["supervisor_dir"]) / "decision.json"
            if decision.exists() and json.loads(decision.read_bytes()).get("phase") == "recovery_hold":
                write_json(
                    run / "root-action-required.json",
                    dict(
                        job_id=job,
                        node=node,
                        supervisor_decision=str(decision),
                        lease_end_utc=plan["lease_end_utc"],
                        srun_pid=step.pid,
                    ),
                )
                return 75
            time.sleep(1)
    after = {k: sha256(k) for k in plan["source_pins"]}
    write_json(run / "source-after.json", after)
    reasons = []
    if step.returncode != 0:
        reasons.append("nonzero dispatch/supervisor exit")
    if after != before:
        reasons.append("source changed")
    if sha256(a.plan) != a.plan_sha256:
        reasons.append("root plan changed")
    sup = Path(plan["supervisor_dir"])
    owner_path = sup / "owner-plan.json"
    decision = json.loads((sup / "decision.json").read_bytes())
    if decision.get("release_lock") is not True or decision.get("owner_exit") != 0:
        reasons.append("owner did not finish and release cleanly")
    if json.loads((sup / "node-source-pins.json").read_bytes()) != before:
        reasons.append("node source pins changed")
    verification = subprocess.run(
        [
            "python3",
            "-B",
            str(HERE / "verify_report.py"),
            str(Path(plan["output"]) / "report.json"),
            "--plan",
            str(owner_path),
        ],
        text=True,
        capture_output=True,
        timeout=300,
    )
    (run / "report-verification.log").write_text(verification.stdout + verification.stderr)
    if verification.returncode:
        reasons.append("independent report verification failed")
    steps = text(["squeue", "--steps", "-h", "-j", job, "-o", "%i|%j|%N"])
    if target_step_lines(steps, node, text):
        reasons.append("node step remained")
    result = dict(
        actual_exit=decision.get("owner_exit"),
        dispatch_exit=step.returncode,
        verified_exit=90 if reasons else 0,
        reasons=reasons,
        source_pin_count=len(before),
        source_maps_equal=before == after,
        clean_close=bool(decision.get("result", {}).get("verified_clean_close")),
        ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        job_id=job,
        node=node,
        scope=plan["scope"],
        native_transfer_tested=False,
        full_model_accepted=False,
    )
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)
    return result["verified_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
