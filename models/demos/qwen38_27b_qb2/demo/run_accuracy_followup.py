# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wait for an owned hardware queue, then run a separate full-GPQA control."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_chunked_prefill_followup import predecessor_ready as clean_predecessor_ready
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.demo.run_overnight_qualification import run


def predecessor_ready(properties, receipt, expected_invocation):
    """A stale completed file never overrides a live controller or a changed job."""
    invocation = properties.get("InvocationID", "")
    if invocation and invocation != expected_invocation:
        raise ValueError("Predecessor service was replaced; refusing a different hardware owner")
    if properties.get("LoadState") not in ("loaded", "not-found"):
        return False
    if properties.get("ActiveState") not in ("inactive", "failed") or properties.get("MainPID") != "0":
        return False
    if properties.get("LoadState") == "loaded":
        if properties.get("Result") != "success":
            raise ValueError("Predecessor service failed; refusing automatic hardware reuse")
    if not receipt or receipt.get("state") != "completed":
        raise ValueError("Predecessor stopped without completing its hardware queue")
    steps = receipt.get("steps", [])
    if (
        len(steps) != 3
        or {step.get("name") for step in steps} != {"native-g0-run", "native-control", "delivery-extended"}
        or any(step.get("state") != "completed" for step in steps)
    ):
        raise ValueError("Predecessor hardware stages did not all finish")
    if receipt.get("native-control", {}).get("owned_processes_stopped") is not True:
        raise ValueError("Predecessor serving workers have not been confirmed stopped")
    # A completed low accuracy score is a valid comparison and does not block
    # the new experiment. Hardware failures and incomplete cleanup do block it.
    return True


def decoder_controls_ready(properties, receipt, expected_invocation):
    """Numerical controls must finish both policies and release devices before G0."""
    if not clean_predecessor_ready(properties, receipt, expected_invocation):
        return False
    runs = receipt.get("runs", [])
    if (
        len(runs) != 2
        or {row.get("name") for row in runs} != {"bfp4-hifi2", "bfp8-hifi2"}
        or any(
            row.get("state") != "completed"
            or row.get("cleanup_completed") is not True
            or len(row.get("logit_metrics", [])) != 8
            for row in runs
        )
    ):
        raise ValueError("Decoder controls did not both complete eight steps and clean up")
    return True


def follow(args):
    if args.status.exists() or args.results.exists():
        raise FileExistsError("Follow-up status and results must be new")
    manifest = json.loads(args.source_manifest.read_text())
    state = dict(
        state="waiting",
        predecessor_unit=args.predecessor_unit,
        predecessor_invocation=args.predecessor_invocation,
        source_manifest_sha256=hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        precision=args.control_precision,
        predecessor_kind=getattr(args, "predecessor_kind", "overnight"),
        survives_disconnect=True,
        resumes_after_reboot=False,
        started_at=time.time(),
    )
    save(args.status, state)
    deadline = time.monotonic() + args.wait_timeout
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Predecessor wait exceeded its bound")
            try:
                result = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.predecessor_unit,
                        "-p",
                        "InvocationID",
                        "-p",
                        "LoadState",
                        "-p",
                        "ActiveState",
                        "-p",
                        "MainPID",
                        "-p",
                        "Result",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                # An observation failure is not evidence that hardware is free.
                state.update(last_observation_error=type(error).__name__, observed_at=time.time())
                save(args.status, state)
                time.sleep(30)
                continue
            properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            receipt = json.loads(args.predecessor_receipt.read_text()) if args.predecessor_receipt.exists() else None
            state.update(last_service=properties, observed_at=time.time())
            save(args.status, state)
            ready = decoder_controls_ready if state["predecessor_kind"] == "decoder-controls" else predecessor_ready
            if ready(properties, receipt, args.predecessor_invocation):
                break
            time.sleep(30)
        for relative, expected in manifest.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen follow-up source changed: {relative}")
        state.update(state="running", hardware_started_at=time.time())
        save(args.status, state)
        # The normal G0 and serving runners still acquire /tmp/tt-device.lock.
        run(
            argparse.Namespace(
                task=args.task,
                source=args.source,
                weights=args.weights,
                results=args.results,
                candidate_g0=None,
                tau_root=None,
                delivery_source=None,
                native_control_only=True,
                control_precision=args.control_precision,
                accuracy_only=True,
            )
        )
        result = json.loads((args.results / "queue.json").read_text())
        state.update(state=result["state"], passed=result["passed"], finished_at=time.time())
    except BaseException as error:
        state.update(state="failed", error=type(error).__name__, detail=str(error)[:2000], finished_at=time.time())
        raise
    finally:
        save(args.status, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "status", "source-manifest", "predecessor-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--predecessor-unit", required=True)
    parser.add_argument("--predecessor-invocation", required=True)
    parser.add_argument("--predecessor-kind", choices=("overnight", "decoder-controls"), default="overnight")
    parser.add_argument("--control-precision", required=True)
    parser.add_argument("--wait-timeout", type=float, default=43200)
    follow(parser.parse_args())
