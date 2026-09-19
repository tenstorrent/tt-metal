# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One GPU-trace comparison with six allocated KV slots; optional loopback copy."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import ttnn
from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4PrefillAdapter, Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_validation import (
    PREPARED_GPU_TRACE_LAYOUT,
    check_table_samples,
    compare_slot_cache,
    read_cache_tensor,
)

GPU_PCC_THRESHOLD = 0.91


@pytest.fixture(params=["mock", "loopback"])
def migration_environment(request, tmp_path):
    gate = request.param
    if gate == "loopback" and os.getenv("GEMMA4_TEST_LOOPBACK") != "1":
        pytest.skip("Start the migration endpoint and set GEMMA4_TEST_LOOPBACK=1")
    trace_dir = Path(os.getenv("PREFILL_TRACE_DIR", Gemma4PrefillAdapter().prefill_trace_default))
    if not (trace_dir / "metadata.json").is_file():
        pytest.skip(f"GPU trace not found: {trace_dir}")
    metadata = json.loads((trace_dir / "metadata.json").read_text())
    assert metadata["model_id"] == Gemma4PrefillAdapter().hf_model_id
    assert metadata["layout"] in ("chunked_group_a_v1", PREPARED_GPU_TRACE_LAYOUT)
    assert metadata["n_layers"] == Gemma4ServiceConfig.NUM_LAYERS
    assert len(metadata["token_ids"]) >= Gemma4ServiceConfig.MAX_SEQ_LEN
    directory = tmp_path
    env = {key: value for key, value in os.environ.items() if not key.startswith("PREFILL_")}
    env.setdefault("OMP_NUM_THREADS", "16")
    env.update(
        {
            key: os.environ[key]
            for key in (
                "PREFILL_TTNN_CACHE",
                "PREFILL_MIGRATION_CLIENT_DIR",
                "PREFILL_MIGRATION_CMD_QUEUE",
                "PREFILL_MIGRATION_TABLE_QUEUE",
                "PREFILL_MIGRATION_RESP_QUEUE",
            )
            if key in os.environ
        }
    )
    env.update(
        PREFILL_MANIFEST=str(Path(__file__).parents[1] / "tt/runners/manifest.json"),
        PREFILL_MODEL="gemma4_d_p",
        PREFILL_SP="8",
        PREFILL_TP="4",
        PREFILL_NUM_LAYERS="60",
        PREFILL_MAX_SEQ_LEN="262144",
        PREFILL_CHUNK_SIZE="8192",
        PREFILL_NUM_USERS=str(Gemma4ServiceConfig.MAX_USER_SLOTS),
        PREFILL_LAYER_ACK_D2H="1",
        PREFILL_H2D_SERVICE_ID=f"gemma4_migration_{gate}_{os.getpid()}",
        PREFILL_MIGRATION_TABLE_PATH=str(directory / "table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(directory / "device_map.json"),
        PREFILL_MOCK_MIGRATION="1" if gate == "mock" else "0",
        PREFILL_ENABLE_MIGRATION="1" if gate == "loopback" else "0",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_ATTACH_WAIT_S="120",
        PREFILL_MIGRATION_WAIT_READY_MS="120000",
        PREFILL_MIGRATION_CMD_QUEUE=env.get("PREFILL_MIGRATION_CMD_QUEUE", "/mig_ep1_cmd"),
        PREFILL_MIGRATION_TABLE_QUEUE=env.get("PREFILL_MIGRATION_TABLE_QUEUE", "/mig_ep1_table"),
        PREFILL_MIGRATION_RESP_QUEUE=env.get("PREFILL_MIGRATION_RESP_QUEUE", "/mig_ep1_resp"),
        PREFILL_TRACE_DIR=str(trace_dir),
    )
    return gate, env


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("context_len", [8192, 16384, 131072, 262144], ids=["8k", "16k", "128k", "256k"])
def test_prefill_migration(migration_environment, context_len, tmp_path):
    gate, env = migration_environment
    env["PREFILL_PCC_SUMMARY_DIR"] = str(tmp_path)
    with (tmp_path / "runner.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.demos.gemma4_d_p.tests.test_prefill_migration",
                gate,
                str(context_len),
                str(tmp_path),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=10800,
        )
    assert result.returncode == 0, (tmp_path / "runner.log").read_text()
    report = json.loads((tmp_path / "gemma4_slot0.json").read_text())
    assert report["slot"] == 0 and report["tokens"] == context_len
    assert len(report["measurements"]) == 1640
    assert min(report["minima"].values()) >= GPU_PCC_THRESHOLD
    if gate == "loopback":
        assert "[migration] WORKER_READY:" in (tmp_path / "runner.log").read_text()
        assert "verify bytes PASSED" in (tmp_path / "producer.log").read_text()


def run_migration_case(gate, context_len, tmp_path):
    env = dict(os.environ)
    from models.demos.common.prefill.runners import prefill_producer, prefill_runner

    client_env = dict(env)
    client_env.update(
        PREFILL_NUM_USERS="1",
        PREFILL_PRODUCER_MAX_REQUESTS="1",
        PREFILL_PRODUCER_CHUNKS=str(context_len // Gemma4ServiceConfig.CHUNK_SIZE),
        PREFILL_PRODUCER_INTERLEAVE="round_robin",
        PREFILL_PRODUCER_CHECK_PCC="0",
        PREFILL_SEND_SHUTDOWN="1",
        PREFILL_H2D_CONNECT_TIMEOUT="120",
        PREFILL_MIGRATION_DEST_ENDPOINT_ID="1",
        PREFILL_MIGRATION_SRC_ENDPOINT_ID="1",
        PREFILL_MIGRATION_PAIRS=f"0:{Gemma4ServiceConfig.MAX_USER_SLOTS - 1}",
        MIGRATION_DONE_FILE=str(tmp_path / "migration_done"),
    )
    module = "prefill_producer" if gate == "mock" else "migration_driver"
    command = [sys.executable, "-m", f"models.demos.common.prefill.runners.{module}"]
    if gate == "loopback":
        command += ["--verify-migration", "dst-bytes"]
    original_loop = prefill_runner.run_request_loop
    failures = []

    def checked_loop(runtime, kv_cache, *args, **kwargs):
        with (tmp_path / "producer.log").open("w") as log:
            producer = subprocess.Popen(command, env=client_env, stdout=log, stderr=subprocess.STDOUT)
            try:
                original_loop(runtime, kv_cache, *args, **kwargs)
                assert producer.wait(timeout=120) == 0, (tmp_path / "producer.log").read_text()
                assert runtime.slot_ends == [context_len] + [0] * (Gemma4ServiceConfig.MAX_USER_SLOTS - 1)
                table = ttnn.experimental.disaggregation.import_from_protobuf_file(env["PREFILL_MIGRATION_TABLE_PATH"])
                device_map = prefill_producer._read_device_map(timeout_s=10)

                def read_heads(layer):
                    cache = kv_cache.layers[layer]
                    tensors = ((0, cache.kv),) if hasattr(cache, "kv") else ((4, cache.k), (20, cache.v))
                    for config_start, tensor in tensors:
                        heads = read_cache_tensor(tensor, 0, context_len)
                        for head, actual in enumerate(heads):
                            config_id = config_start + head
                            check_table_samples(table, device_map, layer, 0, config_id, actual)
                            yield config_id, actual

                scores = compare_slot_cache(read_heads, 0, context_len, env["PREFILL_TRACE_DIR"])
                assert min(scores.values()) >= GPU_PCC_THRESHOLD, scores
            except BaseException as error:
                failures.append(error.with_traceback(None))
            finally:
                if producer.poll() is None:
                    producer.terminate()
                    producer.wait(timeout=30)

    with patch.object(prefill_runner, "run_request_loop", checked_loop):
        prefill_runner.main()
    if failures:
        raise failures[0]


if __name__ == "__main__":
    run_migration_case(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
