# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent candidate/control GPQA, Tau3 and real Galaxy performance queue."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import qualified_groups, verify_qualified_source
from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save


def check_g0(directory, model):
    receipt = json.loads((directory / "full-model.json").read_text())
    qualified_groups(receipt)
    verify_qualified_source(receipt, model)
    suites = ET.parse(directory / "hardware.xml").getroot().findall(".//testsuite")
    if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
        int(s.get(key, 0)) for s in suites for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("Exact-source eight-replica G0 has no clean passing test receipt")


def run(args):
    args.results.mkdir()
    model = args.source / "models/demos/qwen38_27b_qb2"
    status_path = args.results / "queue.json"
    status = dict(
        state="preflight",
        steps=[],
        passed=False,
        hardware_lock="/tmp/tt-device.lock",
        survives_disconnect=True,
        resumes_after_reboot=False,
        scope="Matched candidate/native recurrence at 64K output budget; fixed precision; GPQA, Tau3 and physical Galaxy HTTP sweeps",
    )
    files = [p for p in args.source.rglob("*") if p.is_file() and p.suffix in (".py", ".cpp", ".hpp", ".json", ".sh")]
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status["source_sha256"] = hashes
    save(status_path, status)
    env = environment(args.task, args.source, args.weights)
    for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
        env.pop(key, None)

    def stage(name, command, stage_env, timeout):
        directory = args.results / name
        directory.mkdir()
        row = dict(
            name=name,
            state="running",
            started_at=time.time(),
            command=command,
            timeout_seconds=timeout,
            log=str(directory / "run.log"),
        )
        status["steps"].append(row)
        status.update(state="running", active_step=name)
        save(status_path, status)
        # File-backed logs and controller are owned by the persistent unit.
        # run_capture provides process-group timeouts and disk-growth bounds.
        with (directory / "run.log").open("w") as log:
            # Add a tiny exec wrapper so all descendant output goes to the log.
            wrapper = directory / "exec.py"
            wrapper.write_text(
                "import os,sys\nf=open(sys.argv[1], 'a'); os.dup2(f.fileno(),1); os.dup2(f.fileno(),2)\nos.execvpe(sys.argv[2],sys.argv[2:],os.environ)\n"
            )
            del log
            try:
                run_capture(
                    [str(args.task / "python_env/bin/python"), str(wrapper), str(directory / "run.log"), *command],
                    cwd=args.source,
                    env=stage_env,
                    root=directory,
                    timeout=timeout,
                )
                row.update(state="completed", finished_at=time.time())
            except BaseException as error:
                row.update(state="failed", error=type(error).__name__, finished_at=time.time())
                raise
            finally:
                save(status_path, status)

    try:
        subprocess.run(
            [
                str(args.task / "python_env/bin/python"),
                "-m",
                "pytest",
                str(model / "tests/unit"),
                str(model / "tests/test_benchmark.py"),
                "-c",
                str(args.source / "pytest.ini"),
                f"--rootdir={args.source}",
                "-o",
                "addopts=",
                "-q",
                f"--junitxml={args.results}/unit.xml",
            ],
            cwd=args.source,
            env=env,
            check=True,
            timeout=600,
        )
        tau_env = dict(env, TAU2_DATA_DIR=str(args.tau_root / "tau2/data"))
        tau_env["LD_LIBRARY_PATH"] = (
            str(args.tau_root / "portaudio/root/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        subprocess.run(
            [
                str(args.tau_root / "venv/bin/python"),
                "-c",
                "from tau2.run import run_domain; from models.demos.qwen38_27b_qb2.tests.tau_benchmark import validate_source, preflight; "
                f"from pathlib import Path; validate_source(Path({str(args.tau_root / 'tau2')!r})); preflight(); print('TAU_SOURCE_AND_PREFLIGHT_PASS')",
            ],
            env=tau_env,
            check=True,
            timeout=120,
        )
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen queue source changed: {relative}")
        for label, policy, qualification in (
            ("candidate", "precision_single_step_shared_qk.json", args.candidate_g0),
            ("native-control", "precision_accurate_decode.json", args.results / "native-g0/receipts"),
        ):
            if args.native_control_only and label == "candidate":
                continue
            policy_path = model / "config" / policy
            os.environ["QWEN_PRECISION_CONFIG"] = str(policy_path)
            stage_env = dict(env, QWEN_PRECISION_CONFIG=str(policy_path))
            if label == "native-control":
                qualification.mkdir(parents=True)
                g0_env = dict(
                    stage_env, QWEN_GALAXY_REPLICAS="8", QWEN_GALAXY_RECEIPT=str(qualification / "full-model.json")
                )
                stage(
                    "native-g0-run",
                    [
                        "/bin/bash",
                        str(args.source / "scripts/run_safe_pytest.sh"),
                        str(model / "tests/test_galaxy_replicas.py"),
                        f"--rootdir={args.source}",
                        "-c",
                        str(args.source / "pytest.ini"),
                        "-vv",
                        "-s",
                        "--timeout=5400",
                        f"--junitxml={qualification}/hardware.xml",
                    ],
                    g0_env,
                    6300,
                )
            check_g0(qualification, model)
            command = [
                "/bin/bash",
                str(model / "demo/run_galaxy_serving.sh"),
                str(args.task),
                str(args.results / label / "evaluation"),
                str(qualification / "full-model.json"),
                str(args.source),
                "--exit-after-eval",
                "--sweep-before-exit",
                "--port",
                "8078",
                "--gpqa-max-tokens",
                "65536",
                "--retain-raw-responses",
                "--readiness-timeout",
                "3600",
                "--evaluation-timeout",
                "7200",
            ]
            if label == "candidate" or args.native_control_only:
                command.extend(
                    [
                        "--tau-source",
                        str(args.tau_root / "tau2"),
                        "--tau-python",
                        str(args.tau_root / "venv/bin/python"),
                    ]
                )
            stage(label, command, stage_env, 18000 if label == "candidate" else 14400)
            deployment = json.loads((args.results / label / "evaluation/deployment.json").read_text())
            if deployment.get("state") != "evaluation_completed" or not deployment.get("owned_processes_stopped"):
                raise ValueError("Evaluation or owned-worker cleanup did not complete")
            status[label] = {key: deployment.get(key) for key in ("gpqa", "tau", "passed", "owned_processes_stopped")}
            save(status_path, status)
            # Accuracy failure is a result; it does not cancel the following control.
        delivery_model = args.delivery_source / "models/demos/qwen38_27b_qb2"
        delivery_env = environment(args.task, args.delivery_source, args.weights)
        delivery_env.update(
            QWEN_DRAM_DELIVERY_PROBE="1",
            QWEN_DRAM_DELIVERY_EXTENDED="1",
            QWEN_DRAM_DELIVERY_RECEIPT=str(args.results / "delivery-extended/probe.json"),
        )
        stage(
            "delivery-extended",
            [
                "/bin/bash",
                str(args.delivery_source / "scripts/run_safe_pytest.sh"),
                str(delivery_model / "tests/test_dram_delivery_probe.py"),
                f"--rootdir={args.delivery_source}",
                "-c",
                str(args.delivery_source / "pytest.ini"),
                "-vv",
                "-s",
                "--timeout=1800",
                f"--junitxml={args.results}/delivery-extended/hardware.xml",
            ],
            delivery_env,
            2100,
        )
        tested = "native-control" if args.native_control_only else "candidate"
        status.update(state="completed", passed=bool(status[tested]["passed"]), finished_at=time.time())
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "candidate-g0", "tau-root", "delivery-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--native-control-only", action="store_true", help="Resume unrun control stages in a fresh results directory"
    )
    run(parser.parse_args())
