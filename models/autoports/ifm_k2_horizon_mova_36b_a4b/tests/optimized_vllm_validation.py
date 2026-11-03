# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep one real server for mixed-shape contracts and optional full acceptance."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device-default-top-k", action="store_true")
    parser.add_argument("--qualitative-control", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["VLLM_TT_PLUGIN_ROOT"] = str(Path(env["RUN_ROOT"]) / "vllm-tt-plugin")
    env.update(
        K2_VLLM_BATCH="32",
        K2_VLLM_HOST_COMPAT="1",
        K2_VLLM_EVENTS=str(out / "events.jsonl"),
        TT_METAL_TRACE_ALLOC_TRACKING=str(int(args.tracking)),
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        PYTHONUNBUFFERED="1",
    )
    assert not env.get("K2_VLLM_LAYERS"), "Final evidence requires every model layer"
    assert not any(v not in ("", "0") for k, v in env.items() if "PROFILER" in k or k == "TT_METAL_WATCHER")
    assert not (out / "events.jsonl").exists(), "Use a fresh artifact destination; never reset trace counters"
    commands = []
    launch = ["bash", str(ROOT / "tests/launch_vllm.sh"), "--model-dir", str(out), "--stages", "serve"]
    if args.device_default_top_k:
        # The plugin's mixed-parameter isolation test requires omitted top_k
        # values to remain within the device sampler's supported range.
        launch += [
            "--additional-server-args",
            f"--trust-remote-code --revision {env['HF_REVISION']} "
            f"--tokenizer-revision {env['HF_REVISION']} --max-num-batched-tokens 256 "
            "--override-generation-config '{\"top_k\":32}'",
        ]
    commands.append(launch)
    (out / "environment.json").write_text(
        json.dumps(
            {
                "commands": commands,
                "environment": {k: v for k, v in env.items() if k.startswith(("TT_METAL_", "K2_VLLM_"))},
                "runtime_adapter": str(ROOT / "tt/generator_vllm.py"),
                "artifact_model_dir_only": str(out),
            },
            indent=2,
        )
        + "\n"
    )
    status = {}

    def metrics(name):
        response = requests.get("http://localhost:8000/metrics", timeout=10)
        response.raise_for_status()
        (out / f"metrics_{name}.txt").write_text(response.text)

    def check(name, command):
        commands.append(command)
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        status[name] = result.returncode
        (out / "checks.json").write_text(json.dumps({"commands": commands, "status": status}, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed: {out / (name + '.log')}")

    with (out / "runner.log").open("w") as log:
        server = subprocess.Popen(launch, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}")
                try:
                    if requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Server did not become ready")
            prefix = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."
            if args.qualitative_control:
                check(
                    "qualitative_control",
                    [
                        sys.executable,
                        "-m",
                        prefix + "vllm_qualitative_controls",
                        "--mode",
                        "server",
                        "--control",
                        str(args.qualitative_control.resolve()),
                        "--output",
                        str(out / "qualitative_control.json"),
                    ],
                )
                metrics("after_checks")
                return
            metrics("before_mixed")
            check(
                "mixed_shapes",
                [
                    sys.executable,
                    "-m",
                    prefix + "vllm_mixed_shapes",
                    "--events",
                    str(out / "events.jsonl"),
                    "--output",
                    str(out / "mixed_shapes.json"),
                ],
            )
            metrics("after_mixed")
            check(
                "page_growth",
                [
                    sys.executable,
                    "-m",
                    prefix + "vllm_page_growth_control",
                    "--mixed",
                    str(out / "mixed_shapes.json"),
                    "--output",
                    str(out / "page_growth_control.json"),
                ],
            )
            check(
                "logit_determinism",
                [
                    sys.executable,
                    "-m",
                    prefix + "vllm_logit_determinism",
                    "--output",
                    str(out / "logit_determinism.json"),
                ],
            )
            if args.full:
                if args.device_default_top_k:
                    for repeat in range(3):
                        check(
                            f"isolation_{repeat}",
                            [
                                sys.executable,
                                "-m",
                                "pytest",
                                str(Path(env["VLLM_TT_PLUGIN_ROOT"]) / "tests/tt/test_request_isolation.py"),
                                "-v",
                                "--tt-server-url=http://localhost:8000",
                                "--tt-model-name=IFM/K2-Horizon-MoVA-36B-A4B",
                                "--tt-max-num-seqs=32",
                            ],
                        )
                check(
                    "stress_controls",
                    [
                        sys.executable,
                        "-m",
                        prefix + "vllm_stress_controls",
                        "--output",
                        str(out / "stress_controls.json"),
                    ],
                )
                check(
                    "full_acceptance",
                    [
                        sys.executable,
                        "-m",
                        "readiness_check.run_vllm_server",
                        "--model-dir",
                        str(out),
                        "--hf-model",
                        "IFM/K2-Horizon-MoVA-36B-A4B",
                        "--server-url",
                        "http://localhost:8000",
                        "--stages",
                        "sampling,qualitative",
                        "--max-num-seqs",
                        "32",
                        "--max-model-len",
                        "524288",
                        "--sampling-profile",
                        "full",
                    ],
                )
            metrics("after_checks")
        finally:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            status["server_exit"] = server.returncode
            (out / "checks.json").write_text(json.dumps({"commands": commands, "status": status}, indent=2) + "\n")


if __name__ == "__main__":
    main()
