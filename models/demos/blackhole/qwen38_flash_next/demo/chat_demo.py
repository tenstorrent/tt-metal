# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next end to end on a 1x4 Blackhole mesh: the chat server's start-up acceptance gate, then the
first 64 of the 1,319 GSM8K test items through its OpenAI endpoint.

The launcher starts the server with ``--mtp 4 --acceptance --require-json-96``: the server drafts with the MTP layer,
replays the CPU greedy records and serves only when the ``json`` record matches 96/96. ``tools/benchmark.py`` then
asks the 64 questions greedily (512 output tokens at most, natural EOS) and scores the final ``####`` answers; the
test needs the benchmark's 0.80 floor.
Under ``CI=true``, ``tools/report.py`` writes the standard tt-metal benchmark data.
The server must then stop cleanly on SIGTERM.

Run:  pytest models/demos/blackhole/qwen38_flash_next/demo/chat_demo.py
Env:  MODEL_WEIGHTS_DIR, the checkpoint directory, or HF_MODEL, resolved offline in the local Hugging Face cache at the
      pinned revision; QWEN38_CACHE_ROOT, the launcher's writable cache root; QWEN38_PROFILE, p150-line (default),
      tt-quietbox or qb2; QWEN38_DEVICE_NODES, four KMD device nodes of a larger host for p150-line (e.g. 4,5,6,7);
      GSM8K_TEST_JSONL, the pinned GSM8K test file, downloaded when unset; QWEN38_TT_METAL_SHA, the commit of a tree
      without git history.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION
from models.demos.blackhole.qwen38_flash_next.tools import report
from models.demos.blackhole.qwen38_flash_next.tools.benchmark import DATASET_REVISION

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[3]
LAUNCHER = MODEL_DIR / "tools" / "run_qwen38_chat_server.sh"
GSM8K_URL = (
    f"https://raw.githubusercontent.com/openai/grade-school-math/{DATASET_REVISION}/grade_school_math/data/test.jsonl"
)
DEVICE_NAMES = {"p150-line": "P150x4", "tt-quietbox": "P150x4", "qb2": "P300x2"}  # tools/report.py's device names
MTP_DRAFTS = 4  # the serving form
# About twice the start-up (cold component and JIT caches) and benchmark times on the 1x4 p150 line.
READY_SECONDS = 900
BENCHMARK_SECONDS = 1300
STOP_SECONDS = 60


def log_tail(path: Path, lines: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


@pytest.fixture
def profile() -> str:
    name = os.environ.get("QWEN38_PROFILE", "p150-line")
    assert name in DEVICE_NAMES, f"QWEN38_PROFILE must be one of {sorted(DEVICE_NAMES)}, got {name!r}"
    return name


@pytest.fixture
def checkpoint_dir() -> Path:
    if os.environ.get("MODEL_WEIGHTS_DIR"):
        return Path(os.environ["MODEL_WEIGHTS_DIR"])
    if os.environ.get("HF_MODEL"):
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(os.environ["HF_MODEL"], revision=PINNED_CHECKPOINT_REVISION, local_files_only=True)
        )
    pytest.fail("set MODEL_WEIGHTS_DIR to the checkpoint directory or HF_MODEL to its Hugging Face id")


@pytest.fixture
def chat_server(profile, checkpoint_dir, tmp_path):
    """The server started through the launcher with the acceptance gate: yields its URL, run directory and READY
    record once it serves, then stops it with SIGTERM and requires a clean stop."""

    if not os.environ.get("QWEN38_CACHE_ROOT"):
        pytest.fail("set QWEN38_CACHE_ROOT to the launcher's writable cache root")
    cache_root = Path(os.environ["QWEN38_CACHE_ROOT"]).resolve()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = ["bash", str(LAUNCHER), "--profile", profile, "--checkpoint", str(checkpoint_dir)]
    command += ["--cache-root", str(cache_root), "--python", sys.executable, "--mtp", str(MTP_DRAFTS)]
    command += ["--acceptance", "--require-json-96"]
    command += ["--host", "127.0.0.1", "--port", str(port)]
    if os.environ.get("QWEN38_DEVICE_NODES"):
        command += ["--devices", os.environ["QWEN38_DEVICE_NODES"]]
    # The launcher names the run directory after its pid, which the server keeps (exec). A reused cache root can hold
    # an older run with the same pid, so only a directory made after the launch counts (2 s of slack for coarse file
    # timestamps). The server's own session keeps a terminal's signals away: it sees exactly one SIGTERM.
    server_log = tmp_path / "server.log"
    launched = time.time()
    with server_log.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    run_dir = ready = None
    sent_sigterm = False
    try:
        deadline = time.monotonic() + READY_SECONDS
        while ready is None:
            assert server.poll() is None, f"the server exited {server.returncode} before READY:\n{log_tail(server_log)}"
            assert time.monotonic() < deadline, f"no READY in {READY_SECONDS} s:\n{log_tail(server_log)}"
            time.sleep(5)
            runs = [
                path
                for path in (cache_root / "runs").glob(f"q38-chat-server-*-{server.pid}")
                if path.stat().st_mtime >= launched - 2
            ]
            assert len(runs) <= 1, f"run directories {runs} for one server pid"
            try:
                run_dir = runs[0]
                ready = json.loads((run_dir / "READY").read_text())
            except (IndexError, FileNotFoundError, json.JSONDecodeError):  # not written yet, or mid-write
                continue
        yield f"http://127.0.0.1:{port}", run_dir, ready
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            sent_sigterm = True
            try:
                server.wait(STOP_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    stopped = run_dir / "STOPPED"
    assert stopped.exists(), (
        f"no STOPPED record (SIGTERM sent: {sent_sigterm}, exit {server.returncode}, stop budget {STOP_SECONDS} s):"
        f"\n{log_tail(server_log)}"
    )
    status = json.loads(stopped.read_text())["status"]
    assert status == "stopped", f"the server stopped with status {status}:\n{log_tail(server_log)}"


@run_for_blackhole()
def test_chat_demo(chat_server, profile, checkpoint_dir, tmp_path):
    url, run_dir, ready = chat_server
    acceptance = json.loads((run_dir / "acceptance.json").read_text())
    matched = {record["prompt"]: record["matched_tokens"] for record in acceptance["prompts"]}
    assert ready["acceptance_gate_pass"] and acceptance["gate_pass"], (
        f"acceptance gate: READY {ready['acceptance_gate_pass']}, acceptance.json {acceptance['gate_pass']}; "
        f"matched tokens {matched}"
    )
    logger.debug(f"READY after {ready['startup_seconds']:.0f} s; acceptance records matched {matched}")

    dataset = os.environ.get("GSM8K_TEST_JSONL")
    if not dataset:
        dataset = tmp_path / "gsm8k-test.jsonl"
        with urllib.request.urlopen(GSM8K_URL, timeout=60) as response:
            dataset.write_bytes(response.read())  # the benchmark checks the file's SHA-256
    results = tmp_path / "gsm8k"
    benchmark = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.blackhole.qwen38_flash_next.tools.benchmark",
            "--url",
            url,
            "--checkpoint",
            str(checkpoint_dir),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(results),
            "--limit",
            "64",
        ],
        cwd=REPO_ROOT,
        timeout=BENCHMARK_SECONDS,
    )
    summary_path = results / "summary.json"
    assert summary_path.exists(), f"the benchmark exited {benchmark.returncode} before its summary (captured stdout)"
    summary = json.loads(summary_path.read_text())
    if summary["completed_samples"] == summary["requested_samples"]:
        report.report(summary, device_name=DEVICE_NAMES[profile], mtp_drafts=MTP_DRAFTS)
    assert summary["passed"], (
        f"GSM8K {summary['scope']}: {summary['correct']}/{summary['completed_samples']} correct of "
        f"{summary['requested_samples']}, floor {summary['accuracy_floor']}, benchmark exit {benchmark.returncode}"
    )
    logger.debug(
        f"GSM8K {summary['correct']}/{summary['requested_samples']}, mean TTFT {summary['mean_ttft_s']:.3f} s, "
        f"{summary['mean_stream_decode_tokens_per_s']:.1f} tokens/s"
    )
