# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistently analyze a completed normal or recovered compact profile."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.analyze_operator_profile import run as analyze
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.demo.watch_profile_export import empty_cgroup, properties


def finished(unit, invocation, receipt):
    props = properties(unit)
    if props.get("InvocationID") not in ("", invocation):
        raise ValueError("Profile producer invocation changed")
    if props.get("MainPID") != "0" or props.get("ActiveState") not in ("inactive", "failed"):
        return False
    if props.get("LoadState") not in ("loaded", "not-found"):
        return False
    empty_cgroup(props)
    if props.get("LoadState") == "loaded" and props.get("Result") != "success":
        return False
    return receipt.get("state") == "completed" and receipt.get("cleanup_completed") is True


def run(args):
    args.output.mkdir()
    queue = args.output / "queue.json"
    plan = json.loads(args.plan.read_text())
    status = dict(
        state="waiting",
        physical_devices_accessed=False,
        cleanup_completed=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        started_at=time.time(),
    )

    def terminate(signum, frame):
        raise InterruptedError(f"Profile analysis received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(queue, status)

    def verify_source():
        for name, expected in plan["source_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen operator analysis source changed")

    try:
        verify_source()
        deadline = time.monotonic() + 24 * 3600
        while True:
            selected = None
            for candidate in plan["producers"]:
                receipt_path = Path(candidate["receipt"])
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if finished(candidate["unit"], candidate["invocation"], receipt):
                    root = Path(candidate["capture"])
                    if (root / "analysis.json").is_file():
                        selected = root
                        break
            validation_path = Path(plan["validation"])
            validated = (
                validation_path.is_file() and json.loads(validation_path.read_text()).get("state") == "completed"
            )
            if selected is not None and validated:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("No completed profile available; no restart or hardware takeover")
            time.sleep(20)
        status.update(state="analyzing", capture=str(selected))
        save(queue, status)
        verify_source()
        report = analyze(
            argparse.Namespace(capture=selected, baseline=Path(plan["baseline"]), output=args.output / "report")
        )
        status.update(
            state="completed",
            cleanup_completed=True,
            operation_types=report["distinct_operation_types"],
            device_op_rows=report["device_op_rows_per_rank"],
            unprofiled_step_ms=report["unprofiled_step_ms"],
            profiled_step_ms=report["profiled_step_ms"],
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
