# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent B16 prefill-budget comparison and compact GDN front-end screen."""

import argparse
import hashlib
import json
import math
import signal
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_chunked_prefill_followup import predecessor_ready
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, save_report

POLICY = "precision_single_step_shared_qk_bfp8_all.json"
BUDGETS = (("budget32-before", 32768), ("budget64", 65536), ("budget32-after", 32768))


def summarize_b16(report):
    """Keep measured all-in throughput separate from steady decode rate."""
    if report.get("state") != "completed" or report.get("cleanup_completed") is not True:
        raise ValueError("Require a completed sweep and clean device closure")
    if report.get("replicas") != 1 or report.get("output_tokens") != 128:
        raise ValueError("Require one TP4 and 128 generated tokens")
    cells = report.get("cells", [])
    if [(c.get("input_tokens"), c.get("batch_per_replica")) for c in cells] != [(32768, 16), (16384, 16)]:
        raise ValueError("Require both B16 contexts in priority order")
    rows = []
    for cell in cells:
        samples = cell.get("samples", [])
        if cell.get("status") != "completed" or len(samples) != 3:
            raise ValueError("Require three measured repetitions per context")
        for sample in samples:
            if any(
                not isinstance(sample.get(key), (int, float)) or not math.isfinite(sample[key]) or sample[key] <= 0
                for key in ("prefill_s", "decode_s", "elapsed_s")
            ):
                raise ValueError("Missing positive stage measurements")
            if (
                len(sample.get("ttft_s", [])) != 16
                or any(not math.isfinite(v) or v <= 0 for v in sample["ttft_s"])
                or len(sample.get("output_sha256_per_replica", [])) != 1
            ):
                raise ValueError("Missing per-user TTFT or output identity")
        rows.append(
            dict(
                input_tokens=cell["input_tokens"],
                batch=16,
                output_tokens=128,
                prefill_input_tps=statistics.median(16 * cell["input_tokens"] / s["prefill_s"] for s in samples),
                ttft_s=statistics.median(v for s in samples for v in s["ttft_s"]),
                decode_tsu=statistics.median(127 / s["decode_s"] for s in samples),
                output_tps_including_prefill=statistics.median(16 * 128 / s["elapsed_s"] for s in samples),
                request_elapsed_s=statistics.median(s["elapsed_s"] for s in samples),
                repeatable_tokens=len({s["output_sha256_per_replica"][0] for s in samples}) == 1,
                output_sha256=sorted({s["output_sha256_per_replica"][0] for s in samples}),
            )
        )
    return rows


def passing_test(directory, receipt_name, expected_cases):
    report = json.loads((directory / receipt_name).read_text())
    suites = ET.parse(directory / "hardware.xml").getroot().findall(".//testsuite")
    if (
        report.get("state") != "completed"
        or report.get("passed") is not True
        or report.get("cleanup_completed") is not True
        or len(report.get("cases", [])) != expected_cases
        or sum(int(s.get("tests", 0)) for s in suites) != 1
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise ValueError("Hardware screen lacks complete passing coverage and cleanup")
    for case in report["cases"]:
        if case.get("passed") is not True or compare_timings(case["timings"]) != case["comparison"]:
            raise ValueError("Hardware screen comparison differs from raw evidence")
    return report


def run(args):
    args.output.mkdir()
    path = args.output / "queue.json"
    status = dict(
        state="waiting",
        steps=[],
        primary_batch=16,
        contexts=[32768, 16384],
        predecessor_unit=args.after_unit,
        predecessor_invocation=args.after_invocation,
        survives_disconnect=True,
        resumes_after_reboot=False,
        hardware_lock="/tmp/tt-device.lock",
        promoted_to_serving=False,
        policy=POLICY,
        started_at=time.time(),
    )
    manifest = json.loads(args.manifest.read_text())
    model = args.source / "models/demos/qwen38_27b_qb2"

    def verify_source():
        for name, expected in manifest.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen source changed: " + name)

    def terminate(signum, frame):
        raise InterruptedError(f"B16 follow-up received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(path, status)
    env = environment(args.task, args.source, args.weights)
    for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_VISIBLE_DEVICES", "TT_METAL_DISABLE_SFPLOADMACRO"):
        env.pop(key, None)
    env.update(QWEN_GALAXY_SWEEP="1", QWEN_BATCHED_PREFILL="1", QWEN_PRECISION_CONFIG=str(model / "config" / POLICY))
    python = str(args.task / "python_env/bin/python")

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
            f"--timeout={seconds - 180}",
            f"--junitxml={directory}/hardware.xml",
        ]
        wrapper = directory / "exec.py"
        wrapper.write_text(
            "import os,sys\nf=open(sys.argv[1],'a');os.dup2(f.fileno(),1);os.dup2(f.fileno(),2);os.execvpe(sys.argv[2],sys.argv[2:],os.environ)\n"
        )
        row = dict(name=name, command=command, state="running", started_at=time.time(), timeout_seconds=seconds)
        status["steps"].append(row)
        status.update(state="running", active_stage=name)
        save(path, status)
        try:
            run_capture(
                [python, str(wrapper), str(directory / "run.log"), *command],
                cwd=args.source,
                env=dict(env, **options),
                root=directory,
                timeout=seconds,
            )
            row.update(state="completed", finished_at=time.time())
        except BaseException as error:
            row.update(state="failed", error=type(error).__name__, finished_at=time.time())
            raise
        finally:
            save(path, status)
        return directory

    try:
        verify_source()
        deadline = time.monotonic() + 16 * 3600
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Predecessor remains live; do not interrupt it")
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
                    timeout=30,
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
        # B16 prefill is independent of the unselected front-end prototype.
        # Preserve the currently GPQA-qualified shared-QK policy for all arms.
        summaries = {}
        for name, budget in BUDGETS:
            directory = args.output / name
            plan = make_plan(1, batches=(16,), input_lengths=(32768, 16384))
            plan.update(
                recurrence_variant="shared-qk",
                qualification_scope="B16 prefill budget; no automatic quality qualification",
            )
            save_report(plan, directory)
            stage(
                name,
                "test_galaxy_perf_sweep.py",
                dict(
                    QWEN_SWEEP_RESULTS=str(directory),
                    QWEN_PREFILL_MAX_BATCH_TOKENS=str(budget),
                    QWEN_DECODE_BUCKETS="0",
                ),
                3600,
            )
            summaries[name] = summarize_b16(json.loads((directory / "sweep.json").read_text()))
            save(
                args.output / "prefill-comparison.json",
                dict(
                    arms=summaries,
                    is_serving_benchmark=False,
                    larger_budget_quality_qualified=False,
                    automatic_promotion=False,
                ),
            )
        for name, filename, variable, receipt_name, count in (
            ("preparation-regression", "test_gdn_flat_prepare.py", "QWEN_GDN_FLAT_PREPARE", "prepare.json", 9),
            ("compact-frontend", "test_gdn_frontend.py", "QWEN_GDN_FRONTEND", "frontend.json", 8),
        ):
            directory = args.output / name
            stage(name, filename, {variable: "1", variable + "_RECEIPT": str(directory / receipt_name)}, 1800)
            passing_test(directory, receipt_name, count)
        status.update(state="completed", cleanup_completed=True, finished_at=time.time())
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000], finished_at=time.time())
        raise
    finally:
        save(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "after-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-invocation", required=True)
    run(parser.parse_args())
