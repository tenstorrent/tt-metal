# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native continuation gates; never launches serving or AgentX."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save

PREFIX = "models/demos/qwen38_27b_qb2/"


def validate_stage(measured, layers):
    if (
        measured.get("passed") is not True
        or measured.get("cleanup_completed") is not True
        or measured.get("layers") != layers
        or {row["name"] for row in measured["cases"]} != {"suffix_prefill", "traced_decode", "post_decode_checkpoint"}
    ):
        raise ValueError("Continuation gate lacks complete clean physical evidence")


def run(args):
    args.output.mkdir()
    receipt = args.output / "queue.json"
    report = dict(
        state="preflight",
        passed=False,
        serving_enabled=False,
        agentx_launched=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        started_at=time.time(),
        stages=[],
    )

    def terminate(signum, frame):
        raise InterruptedError(f"Prefix validation received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        manifest = json.loads(args.manifest.read_text())
        for name, expected in manifest.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen source changed: " + name)
        prior = json.loads(args.transfer_receipt.read_text())
        if not all(
            prior.get(k) is True
            for k in (
                "passed",
                "cleanup_completed",
                "all_rank_bytes_exact",
                "neighbours_unchanged",
                "corrupt_restore_unpublished",
            )
        ):
            raise ValueError("Packed transfer has not passed its physical TP4 gate")
        report["transfer_receipt_sha256"] = hashlib.sha256(args.transfer_receipt.read_bytes()).hexdigest()
        # The new numerical test must use the very adapter that passed DMA.
        old_manifest = json.loads((args.transfer_receipt.parent / "source-manifest.json").read_text())
        for name in ("tt/prefix_transfer.py", "tt/prefix_checkpoint.py", "tt/prefix_storage.py"):
            if old_manifest[PREFIX + name] != manifest[PREFIX + name]:
                raise ValueError("Transfer adapter differs from the passed physical test")
        env = environment(args.task, args.source, args.weights)
        for name in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_KERNEL_PATH",
            "TT_VISIBLE_DEVICES",
            "TT_METAL_DISABLE_SFPLOADMACRO",
        ):
            env.pop(name, None)
        env.update(QWEN_DECODE_BUCKETS="1", QWEN_PREFIX_CONTINUATION="1")
        stages = ((4, 1200), (64, 3600))
        if bool(args.reuse_hybrid_receipt) != bool(args.reuse_manifest):
            raise ValueError("Reusing the hybrid gate requires its frozen source manifest")
        if args.reuse_hybrid_receipt:
            prior_manifest = json.loads(args.reuse_manifest.read_text())
            # Only the supervisor/timeout changed. Reuse the already-passed
            # four-layer test only when its model, test and wrapper are exact.
            without_supervisor = lambda values: {
                key: value for key, value in values.items() if key != PREFIX + "demo/run_prefix_validation.py"
            }
            if without_supervisor(prior_manifest) != without_supervisor(manifest):
                raise ValueError("Cannot reuse a hybrid test from different source")
            validate_stage(json.loads(args.reuse_hybrid_receipt.read_text()), 4)
            report["stages"].append(
                dict(
                    layers=4,
                    receipt=str(args.reuse_hybrid_receipt),
                    passed=True,
                    reused=True,
                    sha256=hashlib.sha256(args.reuse_hybrid_receipt.read_bytes()).hexdigest(),
                )
            )
            stages = ((64, 3600),)
        for layers, timeout in stages:
            stage = args.output / f"layers-{layers}"
            stage.mkdir()
            env.update(
                QWEN_PREFIX_LAYERS=str(layers), QWEN_PREFIX_CONTINUATION_RECEIPT=str(stage / "continuation.json")
            )
            command = [
                "/bin/bash",
                str(args.source / "scripts/run_safe_pytest.sh"),
                PREFIX + "tests/test_prefix_continuation.py",
                "-q",
                "-s",
                f"--timeout={timeout - 120}",
                f"--junitxml={stage}/hardware.xml",
            ]
            report.update(state="continuation", layers=layers, timeout_s=timeout)
            save(receipt, report)
            run_capture(command, cwd=args.source, env=env, root=stage, timeout=timeout)
            measured = json.loads((stage / "continuation.json").read_text())
            validate_stage(measured, layers)
            report["stages"].append(dict(layers=layers, receipt=str(stage / "continuation.json"), passed=True))
            save(receipt, report)
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        report["finished_at"] = time.time()
        save(receipt, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "transfer-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-hybrid-receipt", type=Path)
    parser.add_argument("--reuse-manifest", type=Path)
    run(parser.parse_args())
