#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Observe main's unmodified chat benchmark client with eval sampling settings."""
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmark_client import CLIENT, exchange

ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "bringup/artifacts/v5-main-harness"
sys.path.insert(0, str(MAIN))
from llm_module.config import LLMRunConfig, ServerConnection
from llm_module.drivers.vllm import build_vllm_bench_serve_argv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))
from benchmark_phases import socket_path

REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
TOKENIZER = str(Path(os.environ["HF_HOME"]) / "hub/models--google--gemma-4-31B-it/snapshots" / REVISION)


def main():
    args = sys.argv[1:]

    def value(flag):
        return args[args.index(flag) + 1]

    output = Path(value("--result-dir")).resolve()
    name = value("--result-filename")
    batch = int(os.environ.get("BENCHMARK_SERVER_BATCH", value("--max-concurrency")))
    phase_dir = Path(os.environ.get("BENCHMARK_PHASE_DIR", str(output)))
    endpoint = socket_path(phase_dir / f"phase-b{batch}.jsonl")
    measured = "warmup" not in name
    count = int(value("--num-prompts"))
    isl = int(value("--random-input-len"))
    config = LLMRunConfig(
        isl=isl,
        osl=int(value("--random-output-len")),
        max_concurrency=int(value("--max-concurrency")),
        num_prompts=count,
        token_timing=True,
        cache_salt=str(uuid.uuid4()),
    )
    server = ServerConnection(
        base_url=value("--base-url"), service_port=8000, model=value("--model"), tokenizer=TOKENIZER
    )
    command, _ = build_vllm_bench_serve_argv(
        vllm_binary=str(CLIENT),
        config=config,
        server=server,
        result_filename=output / name,
        dataset_seed=int(value("--seed")),
    )
    # Request policy is an explicit benchmark input; main's client source is unchanged.
    extra_index = command.index("--extra-body") + 1
    extra = json.loads(command[extra_index])
    extra.update(chat_template_kwargs={"enable_thinking": True}, temperature=1.0, top_k=20, top_p=0.95)
    if os.environ.get("BENCHMARK_SAMPLING", "gpqa") == "swe":
        extra.pop("seed", None)
    command[extra_index] = json.dumps(extra)
    command += [
        "--ready-check-timeout-sec",
        "0",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
    ]
    (output / (name + ".command.json")).write_text(
        json.dumps(
            {"argv": command, "main_commit": "7a51c2f20f51e7508d8ea365d09092d00bbd5fcb", "sampling": extra}, indent=2
        )
        + "\n"
    )
    if measured:
        start = exchange(endpoint, "reset")
        (output / f"phase-b{batch}-window-start.json").write_text(json.dumps(start) + "\n")
    started = time.monotonic_ns()
    result = subprocess.run(command, env=dict(os.environ, VLLM_PLUGINS=""), check=False)
    ended = time.monotonic_ns()
    if measured:
        end = exchange(endpoint, "drain")
        end.update(client_process_start_ns=started, client_process_end_ns=ended, client_exit_code=result.returncode)
        (output / f"phase-b{batch}-window-end.json").write_text(json.dumps(end) + "\n")
    raw = output / name
    if raw.exists() and not raw.read_bytes().endswith(b"\n"):
        with raw.open("ab") as stream:
            stream.write(b"\n")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
