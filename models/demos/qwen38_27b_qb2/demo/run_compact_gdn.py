# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded compact GDN qualification followed by matched B16 full-model timing."""

import argparse
import hashlib
import json
import signal
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_b16_followup import passing_test, summarize_b16
from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.compact_gdn import (
    BASELINE,
    CANDIDATE,
    CASES,
    COMBINED_GDN_POLICY,
    validate_long_horizon,
)
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import predecessor_ready
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import compare
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, save_report


def validate_layer(report):
    if (
        report.get("state") != "completed"
        or report.get("passed") is not True
        or report.get("cleanup_completed") is not True
        or report.get("candidate_recurrence") != CANDIDATE
    ):
        raise ValueError("Compact GDN real-weight report is not complete and clean")
    rows = report["cases"]
    if [(r["batch"], r["variant"]) for r in rows] != [
        (b, v) for b in (16, 32, 8, 1) for v in ("native", "fused", "native")
    ]:
        raise ValueError("Real-weight coverage differs from frozen plan")
    if [compare(rows[i : i + 3]) for i in range(0, 12, 3)] != report["comparisons"]:
        raise ValueError("Real-weight comparisons differ from raw evidence")
    checks = report.get("changing_input_checks", [])
    if [r.get("batch") for r in checks] != [16, 32] or any(
        r.get("updates") != 64
        or r.get("all_ranks_bit_identical") is not True
        or [c.get("step") for c in r.get("checkpoints", [])] != [1, 2, 4, 8, 16, 32, 64]
        for r in checks
    ):
        raise ValueError("Missing changing-input state/history/projected-output comparison")


def compare_sweeps(arms):
    summaries = {name: summarize_b16(report) for name, report in arms.items()}
    output = []
    for index, context in enumerate((32768, 16384)):
        before, candidate, after = (summaries[k][index] for k in ("before", "compact", "after"))
        drift = abs(after["decode_tsu"] / before["decode_tsu"] - 1)
        tokens_equal = all(r["repeatable_tokens"] for r in (before, candidate, after)) and (
            before["output_sha256"] == candidate["output_sha256"] == after["output_sha256"]
        )
        qualified = drift <= 0.03 and tokens_equal
        baseline = statistics.median([before["decode_tsu"], after["decode_tsu"]])
        output.append(
            dict(
                context=context,
                batch=16,
                arms=[before, candidate, after],
                control_drift_fraction=drift,
                tokens_bit_identical=tokens_equal,
                comparison_qualified=qualified,
                speedup=candidate["decode_tsu"] / baseline if qualified else None,
                target_tsu=30,
                target_reached=qualified and candidate["decode_tsu"] >= 30,
                gpqa_qualified=False,
                promoted_to_serving=False,
            )
        )
    return output


def run(args):
    args.output.mkdir()
    path = args.output / "queue.json"
    status = dict(
        state="waiting",
        steps=[],
        cleanup_completed=False,
        hardware_started=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        promoted_to_serving=False,
        predecessor_unit=args.after_unit,
        predecessor_invocation=args.after_invocation,
        hardware_lock="/tmp/tt-device.lock",
        started_at=time.time(),
        target_tsu=30,
        speculative_decoding=False,
    )
    combined = getattr(args, "combined", False)
    baseline, candidate = (CANDIDATE, COMBINED_GDN_POLICY) if combined else (BASELINE, CANDIDATE)
    status.update(baseline_recurrence=baseline, candidate_recurrence=candidate)
    model = args.source / "models/demos/qwen38_27b_qb2"
    manifest = json.loads(args.manifest.read_text())

    def verify_source():
        for name, digest in manifest.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen compact source changed: " + name)

    def terminate(signum, frame):
        raise InterruptedError(f"Compact GDN queue received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(path, status)
    env = environment(args.task, args.source, args.weights)
    for key in ("TT_METAL_SIMULATOR", "TT_METAL_DISABLE_SFPLOADMACRO", "TT_VISIBLE_DEVICES", "TT_METAL_KERNEL_PATH"):
        env.pop(key, None)

    def stage(name, filename, options, seconds):
        verify_source()
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        command = [
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests" / filename),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            f"--timeout={seconds-180}",
            f"--junitxml={directory}/hardware.xml",
        ]
        row = dict(name=name, state="running", command=command, timeout_seconds=seconds, started_at=time.time())
        status["steps"].append(row)
        status.update(state="running", active_stage=name, hardware_started=True)
        save(path, status)
        try:
            run_capture(command, cwd=args.source, env=dict(env, **options), root=directory, timeout=seconds)
            suites = ET.parse(directory / "hardware.xml").getroot().findall(".//testsuite")
            if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ):
                raise ValueError("Physical test failed or was skipped")
            row.update(state="completed", finished_at=time.time())
        except BaseException as error:
            row.update(state="failed", error=type(error).__name__, finished_at=time.time())
            raise
        finally:
            save(path, status)
        return directory

    try:
        verify_source()
        deadline = time.monotonic() + 28 * 3600
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Predecessor remains live; no restart or takeover")
            try:
                raw = subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.after_unit,
                        "-p",
                        "MainPID",
                        "-p",
                        "ActiveState",
                        "-p",
                        "LoadState",
                        "-p",
                        "Result",
                        "-p",
                        "InvocationID",
                    ],
                    text=True,
                    timeout=20,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                status["observation_error"] = type(error).__name__
                save(path, status)
                time.sleep(20)
                continue
            props = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            receipt = json.loads(args.after_receipt.read_text()) if args.after_receipt.exists() else None
            status["predecessor"] = props
            save(path, status)
            if predecessor_ready(props, receipt, args.after_invocation):
                break
            time.sleep(20)
        if combined:
            directory = stage(
                "combined-4k",
                "test_compact_gdn_long_horizon.py",
                dict(
                    QWEN_COMPACT_LONG_HORIZON="1",
                    QWEN_COMPACT_COMBINED="1",
                    QWEN_COMPACT_LONG_HORIZON_RECEIPT=str(args.output / "combined-4k/long-horizon.json"),
                ),
                1800,
            )
            status["combined_validation"] = validate_long_horizon(
                json.loads((directory / "long-horizon.json").read_text()), baseline=baseline, candidate=candidate
            )
        else:
            directory = stage(
                "epilogue",
                "test_compact_gdn_epilogue.py",
                dict(
                    QWEN_COMPACT_EPILOGUE="1", QWEN_COMPACT_EPILOGUE_RECEIPT=str(args.output / "epilogue/epilogue.json")
                ),
                3600,
            )
            report = passing_test(directory, "epilogue.json", len(CASES))
            if [(r["batch"], r["placement"], tuple(r["mode"])) for r in report["cases"]] != CASES:
                raise ValueError("Compact epilogue coverage differs from frozen plan")
            directory = stage(
                "layer",
                "test_gdn_epilogue_layer.py",
                dict(
                    QWEN_GDN_EPILOGUE_LAYER="1",
                    QWEN_GDN_LAYER_CANDIDATE="compact",
                    QWEN_GDN_LAYER_RECEIPT=str(args.output / "layer/layer.json"),
                ),
                3600,
            )
            validate_layer(json.loads((directory / "layer.json").read_text()))
        arms = {}
        for name, recurrence in (("before", baseline), ("compact", candidate), ("after", baseline)):
            directory = args.output / name
            save_report(make_plan(1, batches=(16,), input_lengths=(32768, 16384)), directory)
            stage(
                name,
                "test_galaxy_perf_sweep.py",
                dict(
                    QWEN_GALAXY_SWEEP="1",
                    QWEN_SWEEP_RESULTS=str(directory),
                    QWEN_BATCHED_PREFILL="1",
                    QWEN_DECODE_BUCKETS="0",
                    QWEN_PREFILL_MAX_BATCH_TOKENS="32768",
                    QWEN_PRECISION_CONFIG=str(model / "config" / f"precision_{recurrence}_bfp8_all.json"),
                ),
                5400,
            )
            arms[name] = json.loads((directory / "sweep.json").read_text())
            summarize_b16(arms[name])
        comparisons = compare_sweeps(arms)
        save(args.output / "comparison.json", dict(rows=comparisons, full_model_measured=True, gpqa_qualified=False))
        # Independently reconcile source/policy/prompt fingerprints and the raw
        # accounting before treating the timed arms as a matched experiment.
        from models.demos.qwen38_27b_qb2.demo.run_compact_followup import measured_win

        winning, _ = measured_win(
            args.output, dict(comparisons=comparisons), manifest, baseline=baseline, candidate=candidate
        )
        status.update(state="completed", cleanup_completed=True, comparisons=comparisons, measured_win=winning)
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "after-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--combined", action="store_true", help="Compare resident state plus compact gates to qualified compact GDN"
    )
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-invocation", required=True)
    run(parser.parse_args())
