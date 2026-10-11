# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qualify a measured compact-GDN win, then attribute its remaining decode cost."""

import argparse
import copy
import hashlib
import json
import math
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_compact_gdn import compare_sweeps
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.demo.run_overnight_qualification import run as qualify
from models.demos.qwen38_27b_qb2.tests.compact_gdn import BASELINE, CANDIDATE, POLICY_PAIRS, policy_pair
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import ARTIFACT_BUDGET, collect
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import predecessor_ready
from models.demos.qwen38_27b_qb2.tests.sweep_recovery import normalized_configuration, resume_measurements

POLICY = "precision_single_step_compact_gdn_bfp8_all.json"
MODEL_PREFIX = "models/demos/qwen38_27b_qb2/"
GPQA_MINIMUM_TSU = 25.0


def gpqa_admission(comparisons):
    """Require measured B16/32K performance, never a shorter-context projection."""
    primary = [row for row in comparisons if row.get("batch") == 16 and row.get("context") == 32768]
    if len(primary) != 1 or primary[0].get("comparison_qualified") is not True:
        raise ValueError("GPQA admission requires one qualified B16/32K comparison")
    tsu = primary[0]["arms"][1]["decode_tsu"]
    if not math.isfinite(tsu) or tsu <= 0:
        raise ValueError("Invalid measured GPQA admission throughput")
    return tsu >= GPQA_MINIMUM_TSU, tsu


def measured_win(directory, receipt, manifest, *, baseline=BASELINE, candidate=CANDIDATE):
    """Recompute the comparison from raw arms; status booleans alone are insufficient."""
    if (baseline, candidate) not in POLICY_PAIRS:
        raise ValueError("Unsupported compact measurement policy pair")
    arms = {name: json.loads((directory / name / "sweep.json").read_text()) for name in ("before", "compact", "after")}
    sources = {
        name.removeprefix(MODEL_PREFIX): digest
        for name, digest in manifest.items()
        if name.startswith(MODEL_PREFIX + "tt/") or name == MODEL_PREFIX + "config/precision.json"
    }
    if not sources:
        raise ValueError("Measured model manifest is empty")
    for name, recurrence in (("before", baseline), ("compact", candidate), ("after", baseline)):
        arm = arms[name]
        policy = MODEL_PREFIX + f"config/precision_{recurrence}_bfp8_all.json"
        if policy not in manifest:
            raise ValueError("Measured policy is missing from the frozen manifest")
        expected = dict(sources, effective_precision_override=manifest[policy])
        if arm.get("source_sha256") != expected or arm["precision"]["decode_recurrence"] != recurrence:
            raise ValueError("Measured source or recurrence differs from frozen compact manifest")
        # Recompute accounting and verify warmup/measurement repeatability.
        resume_measurements(copy.deepcopy(arm), [directory / name / "sweep.json"])
        control = arms["before"]
        for key in ("replicas", "chips", "output_tokens", "warmup_runs", "measured_runs", "methodology"):
            if arm.get(key) != control.get(key):
                raise ValueError("Unmatched full-model measurement protocol: " + key)
        if normalized_configuration(arm["configuration"]) != normalized_configuration(control["configuration"]):
            raise ValueError("Unmatched full-model runtime configuration")
        precision = lambda row: {
            k: v for k, v in row["precision"].items() if k not in ("config_id", "decode_recurrence")
        }
        if precision(arm) != precision(control):
            raise ValueError("Precision changed beyond recurrence selection")
        prompts = lambda row: {(c["input_tokens"], c["batch_per_replica"]): c["prompt_sha256"] for c in row["cells"]}
        if prompts(arm) != prompts(control):
            raise ValueError("Unmatched full-model prompts")
    comparisons = compare_sweeps(arms)
    if comparisons != receipt.get("comparisons"):
        raise ValueError("Compact comparison differs from its raw full-model sweeps")
    if any(not row["comparison_qualified"] for row in comparisons):
        raise ValueError("Compact controls drifted or generated tokens changed")
    # User policy: batch sub-1-TSU changes before expensive G0/API/GPQA.
    # Retain/report the 16K tradeoff; primary admission stays B16/32K.
    before, candidate_arm, after = comparisons[0]["arms"]
    gain_tsu = candidate_arm["decode_tsu"] - (before["decode_tsu"] + after["decode_tsu"]) / 2
    return gain_tsu >= 1.0, comparisons


def verify_model_source(source, compact_manifest):
    """The follow-up may edit harnesses, never the model being qualified."""
    model = source / MODEL_PREFIX
    actual = {}
    for folder in ("tt", "config"):
        for path in (model / folder).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".hpp", ".json"):
                name = str(path.relative_to(source))
                actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {
        name: digest
        for name, digest in compact_manifest.items()
        if name.startswith((MODEL_PREFIX + "tt/", MODEL_PREFIX + "config/"))
    }
    if not expected or actual != expected:
        raise ValueError("Follow-up model differs from measured compact source")


def run(args):
    combined = getattr(args, "combined", False)
    baseline, candidate = policy_pair(combined=combined, padding=getattr(args, "padding", False))
    policy = f"precision_{candidate}_bfp8_all.json"
    args.output.mkdir()
    status_path = args.output / "queue.json"
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
        policy=policy,
        hardware_lock="/tmp/tt-device.lock",
        started_at=time.time(),
    )
    manifest = json.loads(args.manifest.read_text())
    original_manifest = json.loads(args.compact_manifest.read_text())
    model = args.source / "models/demos/qwen38_27b_qb2"

    def verify():
        for name, digest in manifest.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen follow-up source changed: " + name)
        verify_model_source(args.source, original_manifest)

    def terminate(signum, frame):
        raise InterruptedError(f"Compact follow-up received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(status_path, status)
    try:
        verify()
        deadline = time.monotonic() + 24 * 3600
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Compact predecessor still live; no hardware takeover")
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
                save(status_path, status)
                time.sleep(20)
                continue
            props = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            receipt_path = args.compact_results / "queue.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            status["predecessor"] = props
            save(status_path, status)
            if predecessor_ready(props, receipt, args.after_invocation):
                break
            time.sleep(20)
        winning, comparisons = measured_win(
            args.compact_results, receipt, original_manifest, baseline=baseline, candidate=candidate
        )
        status["compact_comparisons"] = comparisons
        gpqa_ready, measured_tsu = gpqa_admission(comparisons)
        status["gpqa_performance_gate"] = dict(
            minimum_tsu=GPQA_MINIMUM_TSU,
            measured_tsu=measured_tsu,
            batch=16,
            context=32768,
            passed=gpqa_ready,
        )
        if not winning or not gpqa_ready:
            status.update(
                state="completed",
                cleanup_completed=True,
                qualification_deferred_for_batch=not winning,
                gpqa_deferred_until_25_tsu=not gpqa_ready,
                skipped_reason=(
                    "Below measured 25 TSU at B16/32K; full GPQA deferred by user policy"
                    if not gpqa_ready
                    else "Less than 1 TSU B16/32K gain; batch before full qualification"
                ),
            )
            return
        verify()
        status.update(state="qualifying", hardware_started=True)
        save(status_path, status)
        qualify(
            argparse.Namespace(
                task=args.task,
                source=args.source,
                weights=args.weights,
                results=args.output / "qualification",
                candidate_g0=None,
                tau_root=None,
                delivery_source=None,
                native_control_only=True,
                control_precision=policy,
                accuracy_only=True,
            )
        )
        qualification = json.loads((args.output / "qualification/queue.json").read_text())
        evaluation = qualification.get("native-control", {})
        if qualification.get("state") != "completed" or evaluation.get("owned_processes_stopped") is not True:
            raise ValueError("Qualification did not complete and release owned serving workers")
        status.update(gpqa=evaluation["gpqa"], accuracy_passed=qualification["passed"])
        save(status_path, status)
        if getattr(args, "skip_profile", False):
            status.update(state="completed", cleanup_completed=True, profile_skipped=True)
            return
        # A completed low score remains useful evidence: keep the diagnostic
        # profile, but never label that policy qualified or promote it.
        args.profile_output.mkdir()
        env = environment(args.task, args.source, args.weights)
        for key in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_KERNEL_PATH",
            "TT_METAL_DISABLE_SFPLOADMACRO",
            "TT_VISIBLE_DEVICES",
            "TT_METAL_PROFILER_NO_CACHE_OP_INFO",
            "TT_METAL_PROFILE_PERF_COUNTERS",
        ):
            env.pop(key, None)
        env.update(
            QWEN_PRECISION_CONFIG=str(model / "config" / policy),
            QWEN_PROFILE_RECURRENCE=candidate,
            QWEN_PROFILE_CONTEXT="32768",
            QWEN_PROFILE_BATCH="16",
            QWEN_FULL_TRACE_PROFILE="1",
            QWEN_DECODE_BUCKETS="0",
            TRACY_NO_WEB_SERVER="1",
            TT_METAL_INSPECTOR_RPC="0",
        )
        for name, profile, seconds in (("unprofiled", False, 1800), ("profiled", True, 5400)):
            verify()
            directory = args.profile_output / name
            directory.mkdir()
            options = dict(env, QWEN_PROFILE_RECEIPT=str(directory / "profile.json"))
            if profile:
                options.update(
                    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="8192",
                    TT_METAL_PROFILER_DIR=str(directory / "tracy"),
                    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES="1",
                )
            command = [
                "/bin/bash",
                str(args.source / "scripts/run_safe_pytest.sh"),
                *(["--profile-ops"] if profile else []),
                str(model / "tests/test_full_trace_profile.py"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-vv",
                "-s",
                "--tb=short",
                f"--timeout={seconds-180}",
                f"--junitxml={directory}/hardware.xml",
            ]
            row = dict(name=name, state="running", command=command, started_at=time.time())
            status["steps"].append(row)
            status.update(state="profiling", active_stage=name)
            save(status_path, status)
            run_capture(
                command, cwd=args.source, env=options, root=directory, timeout=seconds, artifact_budget=ARTIFACT_BUDGET
            )
            report = json.loads((directory / "profile.json").read_text())
            suites = ET.parse(directory / "hardware.xml").getroot().findall(".//testsuite")
            if (
                not report.get("passed")
                or not report.get("cleanup_completed")
                or report.get("precision", {}).get("decode_recurrence") != candidate
                or sum(int(s.get("tests", 0)) for s in suites) != 1
                or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
            ):
                raise ValueError("Full-model compact profile lacks clean exact-policy receipt")
            if profile:
                analysis = collect(directory)
                baseline = json.loads((args.profile_output / "unprofiled/profile.json").read_text())
                if any(report[k] != baseline[k] for k in ("output_hashes", "token_hashes", "operand_hashes")):
                    raise ValueError("Profiling changed inputs or outputs")
                row.update(comparisons=analysis["comparisons"], reconciled=analysis["full_trace_reconciliation_passed"])
            row.update(state="completed", finished_at=time.time())
            save(status_path, status)
        status.update(state="completed", cleanup_completed=True)
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        status["finished_at"] = time.time()
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "task",
        "source",
        "weights",
        "output",
        "profile-output",
        "manifest",
        "compact-manifest",
        "compact-results",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--combined", action="store_true")
    modes.add_argument("--padding", action="store_true")
    parser.add_argument(
        "--skip-profile", action="store_true", help="Run qualification only after the measured gain gate"
    )
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-invocation", required=True)
    run(parser.parse_args())
