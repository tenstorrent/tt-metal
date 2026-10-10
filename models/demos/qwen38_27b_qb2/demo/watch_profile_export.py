# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recover a completed capture and requeue only unstarted dependency failures.

This controller never opens devices or stops another service. Original source,
receipts, captures and launch commands remain unchanged. Replacement experiments
retain their existing hardware lock, source checks, limits and ordering.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.recover_full_profile_export import run as recover
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.profile_export_recovery import (
    replacement_command,
    require_unstarted_failure,
    terminal,
)


def properties(unit):
    command = ["systemctl", "--user", "show", unit]
    for key in ("LoadState", "ActiveState", "MainPID", "InvocationID", "Result", "ControlGroup"):
        command.extend(["-p", key])
    raw = subprocess.check_output(command, text=True, timeout=20)
    return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)


def empty_cgroup(props):
    group = props.get("ControlGroup")
    if not group:
        return
    if not group.startswith("/user.slice/") or ".." in Path(group).parts:
        raise ValueError("Unexpected user-service control group")
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    if any(path.read_text().strip() for path in root.rglob("cgroup.procs")):
        raise ValueError("Original service still owns processes")


def checked_launch(row):
    path = Path(row["launch"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["launch_sha256"]:
        raise ValueError("Original launch changed")
    command = json.loads(raw)["command"]
    # Each retained controller independently rechecks the same manifest before
    # hardware access. Verify it here too, before launching any replacements.
    manifest = Path(command[command.index("--manifest") + 1])
    source = Path(command[command.index("--source") + 1])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != row["manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    for name, digest in json.loads(manifest.read_text()).items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen experiment source changed: " + name)
    return command


def run(args):
    args.output.mkdir()
    queue = args.output / "queue.json"
    plan = json.loads(args.plan.read_text())
    invocation = os.environ.get("INVOCATION_ID")
    if not invocation:
        raise ValueError("A persistent user service with an invocation ID is required")
    status = dict(
        state="waiting",
        cleanup_completed=False,
        hardware_started=False,
        physical_devices_accessed=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        started_at=time.time(),
        replacements=[],
    )

    def terminate(signum, frame):
        raise InterruptedError(f"Export recovery received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(queue, status)
    try:
        deadline = time.monotonic() + 24 * 3600
        while True:
            props = properties(plan["parent_unit"])
            status["parent"] = props
            save(queue, status)
            # Successful transient services may already be garbage-collected.
            # A clean receipt permits only a no-op here, never a replacement.
            if props.get("LoadState") == "not-found" and Path(plan["parent_receipt"]).is_file():
                previous = json.loads(Path(plan["parent_receipt"]).read_text())
                if previous.get("state") == "completed" and previous.get("cleanup_completed") is True:
                    status.update(state="completed", cleanup_completed=True, recovery_needed=False)
                    return
            if terminal(props, plan["parent_invocation"]):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Original controller still live; no takeover")
            time.sleep(20)
        empty_cgroup(props)
        original = json.loads(Path(plan["parent_receipt"]).read_text())
        if props.get("Result") == "success":
            if original.get("state") != "completed" or original.get("cleanup_completed") is not True:
                raise ValueError("Successful parent lacks clean receipt")
            status.update(state="completed", cleanup_completed=True, recovery_needed=False)
            return
        qualification = json.loads(Path(plan["qualification"]).read_text())
        if (
            original.get("state") != "failed"
            or original.get("active_stage") != "profiled"
            or qualification.get("state") != "completed"
            or qualification.get("native-control", {}).get("owned_processes_stopped") is not True
        ):
            raise ValueError("Failure is not a completed qualification followed by profile export")
        validation = json.loads(Path(plan["validation_receipt"]).read_text())
        if validation.get("state") != "completed" or validation.get("full_trace_reconciliation_passed") is not True:
            raise ValueError("CPU export validation has not passed")
        status.update(state="recovering_export", recovery_needed=True)
        save(queue, status)
        recover(
            argparse.Namespace(
                original=Path(plan["profiled"]),
                baseline=Path(plan["baseline"]),
                output=args.output / "report",
                exporter=Path(plan["exporter"]),
                python=Path(plan["python"]),
            )
        )
        # Let the existing dependency checks finish naturally. Never terminate a
        # waiting worker, or restart one that reached a hardware test.
        deadline = time.monotonic() + 240
        while True:
            observed = [(row, properties(row["old_unit"])) for row in plan["followers"]]
            if all(terminal(p, row["old_invocation"]) for row, p in observed):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Followers have not terminated; recovered report retained")
            time.sleep(20)
        commands = []
        for row, observed_props in observed:
            receipt = json.loads(Path(row["old_output"]).joinpath("queue.json").read_text())
            require_unstarted_failure(observed_props, receipt, row["old_invocation"])
            empty_cgroup(observed_props)
            if Path(row["new_output"]).exists() or Path(row["new_control"]).exists():
                raise FileExistsError("Replacement artifacts already exist")
            if properties(row["new_unit"]).get("LoadState") != "not-found":
                raise ValueError("Replacement unit already exists")
            commands.append(checked_launch(row))
        after_unit, after_invocation, after_receipt = plan["unit"], invocation, str(queue)
        for row, command in zip(plan["followers"], commands):
            new_control = Path(row["new_control"])
            new_control.mkdir()
            command = replacement_command(
                command,
                old_unit=row["old_unit"],
                new_unit=row["new_unit"],
                old_output=row["old_output"],
                new_output=row["new_output"],
                old_log=row["old_log"],
                new_log=str(new_control / "run.log"),
                after_unit=after_unit,
                after_invocation=after_invocation,
                after_receipt=after_receipt,
            )
            save(
                new_control / "launch.json",
                dict(command=command, previous_launch=row["launch"], frozen_source_unchanged=True),
            )
            subprocess.run(command, check=True, timeout=30)
            active = properties(row["new_unit"])
            if (
                active.get("ActiveState") != "active"
                or active.get("MainPID") in (None, "0")
                or not active.get("InvocationID")
            ):
                raise ValueError("Replacement did not start")
            save(new_control / "launch-unit.json", active)
            status["replacements"].append(
                dict(unit=row["new_unit"], invocation=active["InvocationID"], output=row["new_output"])
            )
            save(queue, status)
            after_unit, after_invocation, after_receipt = (
                row["new_unit"],
                active["InvocationID"],
                row["new_output"] + "/queue.json",
            )
        # Only a clean exit with this receipt releases the first new experiment.
        status.update(
            state="completed",
            cleanup_completed=True,
            original_failure_preserved=True,
            report=str(args.output / "report"),
        )
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(queue, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
