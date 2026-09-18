# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared producer Gate 1 and shared migration-driver Gate 2 on one Galaxy."""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4ServiceConfig


@contextmanager
def runner_process(env, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "models.demos.gemma4_d_p.tt.runners.prefill_runner"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 1800
            while "setup complete, entering request loop" not in log_path.read_text():
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                if time.monotonic() > deadline:
                    pytest.fail(f"Runner startup timed out: {log_path}")
                time.sleep(1)
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("gate", ["mock", "loopback"])
def test_prefill_migration(gate, tmp_path):
    trace_spec = os.getenv("GEMMA4_MIGRATION_TRACES")
    if not trace_spec:
        pytest.skip("Set GEMMA4_MIGRATION_TRACES to six distinct HF golden trace directories")
    if gate == "loopback" and os.getenv("GEMMA4_TEST_LOOPBACK") != "1":
        pytest.skip("Start the migration endpoint and set GEMMA4_TEST_LOOPBACK=1")
    traces = [Path(path).resolve() for path in trace_spec.split(",")]
    assert len(traces) == Gemma4ServiceConfig.MAX_USER_SLOTS
    for trace in traces:
        for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
            assert (trace / "kv_cache" / f"layer_{layer}.safetensors").is_file()
    prompts = [json.loads((path / "metadata.json").read_text())["token_ids"] for path in traces]
    assert len({tuple(tokens) for tokens in prompts}) == len(traces), "Use distinct prompts to detect crossed slots"
    assert all(8192 <= len(tokens) <= 262144 and len(tokens) % 32 == 0 for tokens in prompts)
    env = {key: value for key, value in os.environ.items() if not key.startswith("PREFILL_")}
    env.update(
        {
            key: os.environ[key]
            for key in (
                "PREFILL_MIGRATION_CLIENT_DIR",
                "PREFILL_MIGRATION_CMD_QUEUE",
                "PREFILL_MIGRATION_TABLE_QUEUE",
                "PREFILL_MIGRATION_RESP_QUEUE",
            )
            if key in os.environ
        }
    )
    if os.getenv("PREFILL_TTNN_CACHE"):
        env["PREFILL_TTNN_CACHE"] = os.environ["PREFILL_TTNN_CACHE"]
    env.update(
        PREFILL_MODEL="gemma4_d_p",
        PREFILL_SP="8",
        PREFILL_TP="4",
        PREFILL_NUM_LAYERS="60",
        PREFILL_MAX_SEQ_LEN="262144",
        PREFILL_CHUNK_SIZE="8192",
        PREFILL_NUM_USERS="6",
        PREFILL_LAYER_ACK_D2H="1",
        PREFILL_H2D_SERVICE_ID=f"gemma4_migration_{gate}_{os.getpid()}",
        PREFILL_MIGRATION_TABLE_PATH=str(tmp_path / "table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(tmp_path / "device_map.json"),
        PREFILL_MOCK_MIGRATION="1" if gate == "mock" else "0",
        PREFILL_ENABLE_MIGRATION="1" if gate == "loopback" else "0",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_ATTACH_WAIT_S="120",
        PREFILL_MIGRATION_WAIT_READY_MS="120000",
        PREFILL_MIGRATION_CMD_QUEUE=env.get("PREFILL_MIGRATION_CMD_QUEUE", "/mig_ep1_cmd"),
        PREFILL_MIGRATION_TABLE_QUEUE=env.get("PREFILL_MIGRATION_TABLE_QUEUE", "/mig_ep1_table"),
        PREFILL_MIGRATION_RESP_QUEUE=env.get("PREFILL_MIGRATION_RESP_QUEUE", "/mig_ep1_resp"),
    )
    client_env = dict(env)
    sources = 6 if gate == "mock" else 3
    client_env.update(
        PREFILL_NUM_USERS=str(sources),
        PREFILL_PRODUCER_MAX_REQUESTS=str(sources),
        PREFILL_PRODUCER_CHUNKS="32",
        PREFILL_PRODUCER_INTERLEAVE="round_robin",
        PREFILL_PRODUCER_SLOT_TRACES=",".join(map(str, traces[:sources])),
        PREFILL_PRODUCER_CHECK_PCC="1",
        PREFILL_STANDALONE_CHUNKED_PCC="0.93",
        PREFILL_SEND_SHUTDOWN="1",
        PREFILL_H2D_CONNECT_TIMEOUT="1800",
        PREFILL_MIGRATION_DEST_ENDPOINT_ID="1",
        PREFILL_MIGRATION_SRC_ENDPOINT_ID="1",
        PREFILL_MIGRATION_PAIRS="0:5,1:3,2:4",
        MIGRATION_DONE_FILE=str(tmp_path / "migration_done"),
    )
    module = "prefill_producer" if gate == "mock" else "migration_driver"
    command = [sys.executable, "-m", f"models.demos.common.prefill.runners.{module}"]
    if gate == "loopback":
        command += ["--verify-migration", "both"]
    with runner_process(env, tmp_path / "runner.log") as runner:
        with (tmp_path / "producer.log").open("w") as log:
            result = subprocess.run(command, env=client_env, stdout=log, stderr=subprocess.STDOUT, timeout=10800)
        assert result.returncode == 0, (tmp_path / "producer.log").read_text()
        assert runner.wait(timeout=120) == 0, (tmp_path / "runner.log").read_text()
    output = (tmp_path / "producer.log").read_text()
    assert "KV cache PCC PASSED" in output
    if gate == "loopback":
        assert "[migration] WORKER_READY:" in (tmp_path / "runner.log").read_text()
        assert "verify bytes PASSED" in output
        assert "verify golden PASSED" in output
