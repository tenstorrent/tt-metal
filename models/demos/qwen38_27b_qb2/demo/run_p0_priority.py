# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Insert bounded P0 work between stages of one identified experiment controller.

Only the parent controller is suspended. Its current child finishes normally.
The safe runner owns the common device lock. Finally and ExecStopPost restore
the same parent, preserving its existing downstream dependencies.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import ARTIFACT_BUDGET, collect


def properties(unit):
    output = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "MainPID", "-p", "InvocationID", "-p", "ActiveState"],
        text=True,
        timeout=30,
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def owned(props, pid, invocation):
    return (
        props.get("MainPID") == str(pid)
        and props.get("InvocationID") == invocation
        and props.get("ActiveState") == "active"
    )


def resume(args):
    # Never signal a replacement process or an unrelated unit.
    if owned(properties(args.controller), args.controller_pid, args.controller_invocation):
        os.kill(args.controller_pid, signal.SIGCONT)


def run(args):
    args.output.mkdir()
    status_path = args.output / "queue.json"
    status = dict(
        state="validating",
        steps=[],
        priority="P0",
        survives_disconnect=True,
        resumes_after_reboot=False,
        controller=args.controller,
        controller_pid=args.controller_pid,
        controller_invocation=args.controller_invocation,
        controller_suspended=False,
        hardware_lock="/tmp/tt-device.lock",
        geometry=dict(batch=16, input_tokens=32768),
        promoted_to_serving=False,
    )
    save(status_path, status)
    manifest = json.loads(args.manifest.read_text())

    def verify():
        for name, digest in manifest.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen P0 source changed: " + name)

    def terminate(signum, frame):
        raise InterruptedError(f"P0 controller received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        verify()
        if not owned(properties(args.controller), args.controller_pid, args.controller_invocation):
            raise ValueError("Experiment controller identity changed")
        before = json.loads(args.controller_receipt.read_text())
        if before.get("active_stage") not in ("native-before", "shared-qk", "native-after"):
            raise ValueError("Controller is not in a bounded performance sweep")
        stage = before["active_stage"]
        os.kill(args.controller_pid, signal.SIGSTOP)
        status.update(controller_suspended=True, state="waiting_for_current_sweep", current_sweep=stage)
        save(status_path, status)
        after = json.loads(args.controller_receipt.read_text())
        if after.get("active_stage") != stage:
            raise ValueError("Controller advanced while reserving priority")
        receipt_path = args.controller_receipt.parent / stage / "sweep.json"
        deadline = time.monotonic() + 4200
        while True:
            report = json.loads(receipt_path.read_text())
            if report.get("cleanup_completed"):
                if report.get("state") != "completed":
                    raise ValueError("Current sweep failed; no automatic P0 hardware start")
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Current sweep did not release hardware in 70 minutes")
            time.sleep(5)
        env = environment(args.task, args.source, args.weights)
        model = args.source / "models/demos/qwen38_27b_qb2"
        env.update(
            QWEN_FULL_TRACE_PROFILE="1",
            QWEN_PROFILE_CONTEXT="32768",
            QWEN_PROFILE_BATCH="16",
            QWEN_PRECISION_CONFIG=str(model / "config/precision_single_step_shared_qk_bfp8_all.json"),
            TRACY_NO_WEB_SERVER="1",
            TT_METAL_INSPECTOR_RPC="0",
        )
        for key in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_KERNEL_PATH",
            "TT_VISIBLE_DEVICES",
            "TT_METAL_DISABLE_SFPLOADMACRO",
            "TT_METAL_PROFILER_NO_CACHE_OP_INFO",
            "TT_METAL_PROFILE_PERF_COUNTERS",
        ):
            env.pop(key, None)
        python = str(args.task / "python_env/bin/python")
        for name, profile, seconds in (("unprofiled", False, 1800), ("profiled", True, 5400)):
            verify()
            directory = args.output / name
            directory.mkdir()
            case_env = dict(env, QWEN_PROFILE_RECEIPT=str(directory / "profile.json"))
            if profile:
                # Default capacity is 1000 programs, below this complete graph.
                # Both the host and JIT read this setting; no native build edit.
                case_env.update(
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
                f"--timeout={seconds - 180}",
                f"--junitxml={directory}/hardware.xml",
            ]
            wrapper = directory / "exec.py"
            wrapper.write_text(
                "import os,sys\nf=open(sys.argv[1],'a');os.dup2(f.fileno(),1);"
                "os.dup2(f.fileno(),2);os.execvpe(sys.argv[2],sys.argv[2:],os.environ)\n"
            )
            row = dict(name=name, state="running", command=command, started_at=time.time())
            status["steps"].append(row)
            status.update(state="running", active_stage=name)
            save(status_path, status)
            run_capture(
                [python, str(wrapper), str(directory / "run.log"), *command],
                cwd=args.source,
                env=case_env,
                root=directory,
                timeout=seconds,
                artifact_budget=ARTIFACT_BUDGET,
            )
            receipt = json.loads((directory / "profile.json").read_text())
            if not (receipt.get("passed") and receipt.get("cleanup_completed")):
                raise ValueError("Full-model diagnostic lacks passing clean receipt")
            row.update(state="completed", finished_at=time.time(), replays=receipt["replays"])
            if profile:
                analysis = collect(directory)
                row["full_trace_reconciliation_passed"] = analysis["full_trace_reconciliation_passed"]
                row["comparisons"] = analysis["comparisons"]
                baseline = json.loads((args.output / "unprofiled/profile.json").read_text())
                if (
                    receipt["output_hashes"] != baseline["output_hashes"]
                    or receipt["token_hashes"] != baseline["token_hashes"]
                ):
                    raise ValueError("Profiling changed full-model output")
            save(status_path, status)
        status.update(state="completed", cleanup_completed=True)
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        resume(args)
        status["controller_suspended"] = False
        status["resume_attempted"] = True
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "output", "manifest", "controller-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--controller-pid", required=True, type=int)
    parser.add_argument("--controller-invocation", required=True)
    parser.add_argument("--resume-only", action="store_true")
    arguments = parser.parse_args()
    resume(arguments) if arguments.resume_only else run(arguments)
