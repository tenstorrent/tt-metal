# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run corrected GPQA first, public OpenBench second, then bounded BFP8 sweeps."""

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.overnight_plan import load_followup, perf_cases, remaining_stage_seconds
from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.demo.run_overnight_qualification import check_g0
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, save_report


def run(args):
    root, source, task = args.output, args.source, args.task
    root.mkdir(exist_ok=False)
    model = source / "models/demos/qwen38_27b_qb2"
    env = environment(task, source, args.weights)
    env["QWEN_PRECISION_CONFIG"] = str(model / "config/precision_accurate_decode_bfp8_all.json")
    os.environ["QWEN_PRECISION_CONFIG"] = env["QWEN_PRECISION_CONFIG"]
    for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_VISIBLE_DEVICES"):
        env.pop(key, None)
    state = dict(
        state="preflight",
        started_at=time.time(),
        budget_seconds=28800,
        stages=[],
        survives_disconnect=True,
        resumes_after_reboot=False,
        hardware_lock="/tmp/tt-device.lock",
    )
    status = root / "queue.json"
    deadline = time.monotonic() + 28800
    save(status, state)

    def terminate(signum, frame):
        raise InterruptedError(f"Queue received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def stage(name, command, stage_env, seconds):
        directory = root / name
        directory.mkdir(exist_ok=True)
        wrapper = directory / "exec.py"
        wrapper.write_text(
            "import os,sys\nf=open(sys.argv[1],'a');os.dup2(f.fileno(),1);os.dup2(f.fileno(),2);os.execvpe(sys.argv[2],sys.argv[2:],os.environ)\n"
        )
        row = dict(name=name, command=command, timeout_seconds=seconds, state="running", started_at=time.time())
        state["stages"].append(row)
        state.update(state="running", active_stage=name)
        save(status, state)
        try:
            run_capture(
                [str(task / "python_env/bin/python"), str(wrapper), str(directory / "run.log"), *command],
                cwd=source,
                env=stage_env,
                root=directory,
                timeout=seconds,
            )
            row.update(state="completed", finished_at=time.time())
        except BaseException as error:
            row.update(state="failed", error=type(error).__name__, detail=str(error)[:2000], finished_at=time.time())
            raise
        finally:
            save(status, state)

    try:
        for relative, expected in json.loads(args.manifest.read_text()).items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen source changed: " + relative)
        check_g0(args.qualification.parent, model)
        load_followup(args.followup)
        command = [
            "/bin/bash",
            str(model / "demo/run_galaxy_serving.sh"),
            str(task),
            str(root / "qualification/evaluation"),
            str(args.qualification),
            str(source),
            "--exit-after-eval",
            "--port",
            "8078",
            "--gpqa-max-tokens",
            "65536",
            "--retain-raw-responses",
            "--readiness-timeout",
            "3600",
            "--evaluation-timeout",
            "7200",
            "--followup-command",
            str(args.followup),
        ]
        stage("qualification", command, env, 18480)
        report = json.loads((root / "qualification/evaluation/deployment.json").read_text())
        if report.get("state") != "evaluation_completed" or report.get("owned_processes_stopped") is not True:
            raise ValueError("Qualification process did not finish and clean up")
        state.update(gpqa=report["gpqa"], openbench_exit_code=report.get("followup_exit_code"))
        save(status, state)
        # A measured accuracy miss is a retained result. A hardware/process
        # failure aborts the queue; do not recycle unhealthy devices blindly.
        for case in perf_cases():
            seconds = remaining_stage_seconds(deadline, time.monotonic(), case["seconds"])
            if not seconds:
                state.setdefault("not_run_budget", []).append(case["name"])
                continue
            directory = root / case["name"]
            batches = tuple(sorted({b for _, b in case["cells"]}))
            lengths = tuple(dict.fromkeys(n for n, _ in case["cells"]))
            plan = make_plan(1, batches=batches, input_lengths=lengths)
            cells = {(c["input_tokens"], c["batch_per_replica"]): c for c in plan["cells"]}
            plan["cells"] = [cells[pair] for pair in case["cells"]]
            assert all(c["status"] == "queued" for c in plan["cells"])
            save_report(plan, directory)
            perf_env = dict(
                env, QWEN_SWEEP_RESULTS=str(directory), QWEN_PREFILL_MAX_BATCH_TOKENS=str(case["token_budget"])
            )
            command = [
                "/bin/bash",
                str(source / "scripts/run_safe_pytest.sh"),
                str(model / "tests/test_galaxy_perf_sweep.py"),
                f"--rootdir={source}",
                "-c",
                str(source / "pytest.ini"),
                "-vv",
                "-s",
                f"--timeout={max(60, seconds - 180)}",
                f"--junitxml={directory}/hardware.xml",
            ]
            stage(case["name"], command, perf_env, seconds)
            measured = json.loads((directory / "sweep.json").read_text())
            if measured.get("state") != "completed" or measured.get("cleanup_completed") is not True:
                raise ValueError("Performance sweep did not complete and release devices")
        state.update(state="completed", cleanup_completed=True, finished_at=time.time())
    except BaseException as error:
        state.update(state="failed", error=type(error).__name__, detail=str(error)[:2000], finished_at=time.time())
        raise
    finally:
        save(status, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "source", "task", "weights", "manifest", "qualification", "followup"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
