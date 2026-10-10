# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent matched BFP8 recurrence comparison, profiles and full GPQA."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.bounded_profile import collect
from models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps import compare, render
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, save_report
from models.demos.qwen38_27b_qb2.tt.precision import load_precision

POLICIES = {
    "native": "precision_accurate_decode_bfp8_all.json",
    "shared-qk": "precision_single_step_shared_qk_bfp8_all.json",
}
FUSION_POLICIES = {
    "native": "precision_single_step_shared_qk_bfp8_all.json",
    "shared-qk": "precision_single_step_flat_prepare_epilogue_bfp8_all.json",
}


def predecessor_ready(properties, receipt, invocation, *, clean_release=False, layer_release=False):
    if properties.get("InvocationID") not in ("", invocation):
        raise ValueError("Predecessor invocation changed")
    if properties.get("MainPID") != "0" or properties.get("ActiveState") not in ("inactive", "failed"):
        return False
    if properties.get("LoadState") not in ("loaded", "not-found"):
        return False
    if properties.get("LoadState") == "loaded" and properties.get("Result") != "success":
        raise ValueError("Predecessor exited unsuccessfully; inspect before using hardware")
    if layer_release:
        from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import BATCHES, compare

        if (
            not receipt
            or receipt.get("state") != "completed"
            or receipt.get("passed") is not True
            or receipt.get("cleanup_completed") is not True
            or receipt.get("candidate_recurrence") != "single_step_flat_prepare_epilogue"
        ):
            raise ValueError("Predecessor lacks passing flat-preparation layer qualification")
        cases = receipt.get("cases", [])
        if len(cases) != 3 * len(BATCHES):
            raise ValueError("Incomplete layer qualification")
        comparisons = [compare(cases[i : i + 3]) for i in range(0, len(cases), 3)]
        if tuple(c["batch"] for c in comparisons) != BATCHES or comparisons != receipt.get("comparisons"):
            raise ValueError("Layer comparison receipts disagree")
        return True
    if clean_release:
        if (
            not receipt
            or receipt.get("state") != "completed"
            or receipt.get("cleanup_completed") is not True
            or receipt.get("owned_container_removed") is not True
            or receipt.get("device_reset_required") is not True
            or receipt.get("hardware_health_proven") is not False
            or receipt.get("reason") != "performance_priority"
        ):
            raise ValueError("Predecessor lacks audited performance-priority release")
        return True
    if (
        not receipt
        or receipt.get("state") != "completed"
        or receipt.get("passed") is not True
        or receipt.get("api_and_tool_smoke_passed") is not True
        or receipt.get("owned_container_removed") is not True
        or receipt.get("evaluation_exit_code") != 0
    ):
        raise ValueError("Predecessor lacks successful qualification and owned-container cleanup")
    return True


def validate_policies(model, *, fusion=False):
    selected = FUSION_POLICIES if fusion else POLICIES
    policies = [load_precision(model / "config" / name) for name in selected.values()]
    expected = (
        ("single_step_shared_qk", "single_step_flat_prepare_epilogue")
        if fusion
        else ("native", "single_step_shared_qk")
    )
    if tuple(p["decode_recurrence"] for p in policies) != expected:
        raise ValueError("Wrong recurrence selection")
    if {k: v for k, v in policies[0].items() if k not in ("config_id", "decode_recurrence")} != {
        k: v for k, v in policies[1].items() if k not in ("config_id", "decode_recurrence")
    }:
        raise ValueError("Comparison changes precision beyond recurrence selection")
    if (
        set(policies[0]["weight_groups"].values()) != {"bfloat8_b"}
        or policies[0]["kv_cache_dtype"] != "bfloat8_b"
        or policies[0]["recurrent_dtype"] != "float32"
    ):
        raise ValueError("Require BFP8 weights/KV and FP32 recurrent state")


def control_stable(report, tolerance=3.0):
    return bool(report["cells"]) and all(
        row.get("same_output_hash_as_native") is True
        and abs(row.get("decode_uplift_percent", float("inf"))) <= tolerance
        for row in report["cells"]
    )


def run(args):
    args.output.mkdir()
    status_path = args.output / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    selected = FUSION_POLICIES if args.fusion else POLICIES
    recurrence_policies = tuple(
        load_precision(model / "config" / name)["decode_recurrence"] for name in selected.values()
    )
    status = dict(
        state="waiting",
        steps=[],
        predecessor_unit=args.after_unit,
        predecessor_invocation=args.after_invocation,
        hardware_started=False,
        hardware_lock="/tmp/tt-device.lock",
        survives_disconnect=True,
        resumes_after_reboot=False,
        promoted_to_serving=False,
        started_at=time.time(),
        profiles_first=args.profiles_first,
        epilogue_first=args.epilogue_first,
        release_audit=args.after_clean_release,
        variant_configs=selected,
        fusion_followup=args.fusion,
    )
    manifest = json.loads(args.manifest.read_text())

    def verify_source():
        for relative, digest in manifest.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen source changed: " + relative)

    def terminate(signum, frame):
        raise InterruptedError(f"Follow-up received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    save(status_path, status)
    env = environment(args.task, args.source, args.weights)
    for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_VISIBLE_DEVICES", "TT_METAL_DISABLE_SFPLOADMACRO"):
        env.pop(key, None)
    python = str(args.task / "python_env/bin/python")
    runner = args.source / "scripts/run_safe_pytest.sh"

    def stage(name, command, stage_env, seconds):
        verify_source()
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        wrapper = directory / "exec.py"
        wrapper.write_text(
            "import os,sys\nf=open(sys.argv[1],'a');os.dup2(f.fileno(),1);os.dup2(f.fileno(),2);os.execvpe(sys.argv[2],sys.argv[2:],os.environ)\n"
        )
        row = dict(name=name, command=command, timeout_seconds=seconds, state="running", started_at=time.time())
        status["steps"].append(row)
        status.update(state="running", active_stage=name, hardware_started=True)
        save(status_path, status)
        try:
            run_capture(
                [python, str(wrapper), str(directory / "run.log"), *command],
                cwd=args.source,
                env=stage_env,
                root=directory,
                timeout=seconds,
            )
            row.update(state="completed", finished_at=time.time())
        except BaseException as error:
            row.update(state="failed", error=type(error).__name__, finished_at=time.time())
            raise
        finally:
            save(status_path, status)

    def test_command(test, directory, seconds, profile=False):
        return [
            "/bin/bash",
            str(runner),
            *(["--profile-ops"] if profile else []),
            str(model / "tests" / test),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            f"--timeout={seconds - 180}",
            f"--junitxml={directory}/hardware.xml",
        ]

    def profiles():
        for batch in (32, 16):
            for label, recurrence in zip(selected, recurrence_policies):
                name = f"profile-s32768-b{batch}-{label}"
                directory = args.output / name
                profile_env = dict(
                    env,
                    QWEN_PRECISION_CONFIG=str(model / "config" / selected[label]),
                    QWEN_BOUNDED_LAYER_PROFILE="1",
                    QWEN_PROFILE_CONTEXT="32768",
                    QWEN_PROFILE_BATCH=str(batch),
                    QWEN_PROFILE_RECURRENCE=recurrence,
                    QWEN_PROFILE_RECEIPT=str(directory / "profile.json"),
                    TT_METAL_PROFILER_DIR=str(directory / "tracy"),
                    TRACY_NO_WEB_SERVER="1",
                    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES="1",
                )
                stage(name, test_command("test_bounded_layer_profile.py", directory, 1200, True), profile_env, 1200)
                collect(directory, 32768, batch, recurrence)

    def epilogue_diagnostic():
        directory = args.output / "epilogue"
        stage_env = dict(env, QWEN_GDN_EPILOGUE="1", QWEN_GDN_EPILOGUE_RECEIPT=str(directory / "epilogue.json"))
        stage("epilogue", test_command("test_gdn_epilogue.py", directory, 5400), stage_env, 5400)
        report = json.loads((directory / "epilogue.json").read_text())
        if (
            report.get("state") != "completed"
            or report.get("passed") is not True
            or report.get("cleanup_completed") is not True
        ):
            raise ValueError("Epilogue diagnostic failed or did not release hardware")

    try:
        verify_source()
        validate_policies(model, fusion=args.fusion)
        deadline = time.monotonic() + args.wait_timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Predecessor wait expired; no hardware opened")
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
                save(status_path, status)
                time.sleep(20)
                continue
            properties = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            receipt = json.loads(args.after_receipt.read_text()) if args.after_receipt.exists() else None
            status["predecessor"] = properties
            save(status_path, status)
            if predecessor_ready(
                properties,
                receipt,
                args.after_invocation,
                clean_release=args.after_clean_release,
                layer_release=args.fusion,
            ):
                break
            time.sleep(20)
        if args.fusion:
            # Re-run the short layer gate on this exact frozen source, including
            # formatting-only differences from the exploratory hardware run.
            directory = args.output / "flat-layer"
            stage_env = dict(
                env,
                QWEN_GDN_EPILOGUE_LAYER="1",
                QWEN_GDN_LAYER_CANDIDATE="flat_prepare",
                QWEN_GDN_LAYER_RECEIPT=str(directory / "layer.json"),
                QWEN_PRECISION_CONFIG=str(model / "config" / selected["shared-qk"]),
            )
            stage("flat-layer", test_command("test_gdn_epilogue_layer.py", directory, 1200), stage_env, 1200)
            layer_report = json.loads((directory / "layer.json").read_text())
            predecessor_ready(
                dict(MainPID="0", ActiveState="inactive", LoadState="not-found", InvocationID=""),
                layer_report,
                "",
                layer_release=True,
            )
        if args.profiles_first:
            profiles()
        if args.epilogue_first:
            epilogue_diagnostic()
        # Each safe runner obtains the same global flock; the controller never
        # holds it while waiting on a child that also needs the lock.
        for name, policy in (("native-before", "native"), ("shared-qk", "shared-qk"), ("native-after", "native")):
            directory = args.output / name
            plan = make_plan(1, batches=(32, 16), input_lengths=(32768, 16384))
            plan.update(
                recurrence_variant=policy, qualification_scope="Matched BFP8 throughput; separate full GPQA follows"
            )
            save_report(plan, directory)
            stage_env = dict(
                env, QWEN_SWEEP_RESULTS=str(directory), QWEN_PRECISION_CONFIG=str(model / "config" / selected[policy])
            )
            stage(name, test_command("test_galaxy_perf_sweep.py", directory, 3600), stage_env, 3600)
            measured = json.loads((directory / "sweep.json").read_text())
            if measured.get("state") != "completed" or measured.get("cleanup_completed") is not True:
                raise ValueError("Sweep did not complete and release devices")
        controls = compare(
            args.output / "native-before/sweep.json",
            args.output / "native-after/sweep.json",
            variants=("native", "native"),
            recurrence_policies=(recurrence_policies[0], recurrence_policies[0]),
            require_same_output=True,
        )
        status["native_control_stable_within_3_percent"] = control_stable(controls)
        render(controls, args.output / "control-drift")
        for label in ("native-before", "native-after"):
            comparison = compare(
                args.output / label / "sweep.json",
                args.output / "shared-qk/sweep.json",
                variants=("native", "shared-qk"),
                recurrence_policies=recurrence_policies,
                require_same_output=args.fusion,
            )
            render(comparison, args.output / ("comparison-" + label))
        if not args.profiles_first and not args.fusion:
            profiles()
        if args.qualify:
            # Reuse the accuracy-only runner's exact-source G0 and complete GPQA.
            # Its historical "native-control" label is generic here; the policy
            # and worker receipts identify the optimized BFP8 candidate explicitly.
            command = [
                python,
                "-m",
                "models.demos.qwen38_27b_qb2.demo.run_overnight_qualification",
                "--task",
                str(args.task),
                "--source",
                str(args.source),
                "--weights",
                str(args.weights),
                "--results",
                str(args.output / "candidate-qualification/results"),
                "--candidate-g0",
                str(args.output / "candidate-qualification/results/native-g0/receipts"),
                "--tau-root",
                str(args.task / "tau3"),
                "--delivery-source",
                str(args.source),
                "--native-control-only",
                "--accuracy-only",
                "--control-precision",
                selected["shared-qk"],
            ]
            stage("candidate-qualification", command, env, 21600)
            evaluation = json.loads(
                (args.output / "candidate-qualification/results/native-control/evaluation/deployment.json").read_text()
            )
            if (
                evaluation.get("state") != "evaluation_completed"
                or evaluation.get("owned_processes_stopped") is not True
            ):
                raise ValueError("Candidate GPQA did not complete and stop owned workers")
            status["candidate_gpqa"] = evaluation["gpqa"]
        status.update(state="completed", cleanup_completed=True, finished_at=time.time())
    except BaseException as error:
        status.update(state="failed", error=type(error).__name__, detail=str(error)[:2000], finished_at=time.time())
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "after-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-invocation", required=True)
    parser.add_argument("--wait-timeout", type=int, default=64800)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--after-clean-release", action="store_true")
    parser.add_argument("--profiles-first", action="store_true")
    parser.add_argument("--epilogue-first", action="store_true")
    parser.add_argument(
        "--fusion", action="store_true", help="Compare qualified shared Q/K against direct preparation plus epilogue"
    )
    args = parser.parse_args()
    if args.fusion and (args.epilogue_first or args.after_clean_release):
        parser.error("Fusion follow-up uses its real-weight layer gate, without legacy epilogue/release flags")
    run(args)
