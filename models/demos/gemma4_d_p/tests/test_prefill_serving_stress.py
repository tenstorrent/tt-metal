# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill serving stress test: the production prefill runner (traced model, H2D request service, per-layer acks)
serves a long randomized multi-user workload from the producer, so prefill runs repeatedly the way it does in serving.

The producer draws request lengths, interleaving, mid-chunk ends and multi-turn continuations from a seeded RNG and
runs until it has issued GEMMA4_STRESS_REQUESTS requests or GEMMA4_STRESS_DURATION_S seconds have passed. The test
checks that every chunk is served and acknowledged for every layer, that the output stays finite, and that per-chunk
latency at a given context depth does not drift over the run.

Knobs (environment):
    GEMMA4_STRESS_CHUNK_SIZE   prefill chunk size (default 8192)
    GEMMA4_STRESS_REQUESTS     requests to issue (default 48)
    GEMMA4_STRESS_DURATION_S   wall-clock budget for issuing requests (default inf)
    GEMMA4_STRESS_CHUNKS       "min,max" chunks per request (default "1,<max>", max = 256k context / chunk size)
    GEMMA4_STRESS_SEED         workload seed (default 1234)
    GEMMA4_STRESS_GAP_PROB     probability of an idle gap between chunks (default 0)
    GEMMA4_STRESS_CHECK_EVERY  check output health every N chunks (default 16)
    GEMMA4_STRESS_MAX_DRIFT    allowed late / early median chunk latency at the same context depth (default 1.25)
"""

import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime

_MAX_SEQ_LEN = 262144
_NUM_USERS = 6
# Chunk latency grows with context depth (global attention reads the whole prefix), so drift is compared per bucket.
_DEPTH_BUCKET_TOKENS = 32768


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _write_producer_manifest(path, chunk_size):
    manifest = json.loads((Path(__file__).parents[1] / "tt/runners/manifest.json").read_text())
    manifest["env"]["PREFILL_CHUNK_SIZE"] = str(chunk_size)
    manifest["env"]["PREFILL_NUM_USERS"] = str(_NUM_USERS)
    max_chunks = _MAX_SEQ_LEN // chunk_size
    manifest["transport"] = {"connect_timeout_s": 1200}
    manifest["workload"] = {
        "num_users": _NUM_USERS,
        "chunks": os.environ.get("GEMMA4_STRESS_CHUNKS", f"1,{max_chunks}"),
        "max_requests": _env_int("GEMMA4_STRESS_REQUESTS", 48),
        "duration_s": os.environ.get("GEMMA4_STRESS_DURATION_S", "inf"),
        "interleave": "random",
        "p_gap": float(os.environ.get("GEMMA4_STRESS_GAP_PROB", "0")),
        "mid_end_prob": 0.3,
        # The producer starts a follow-up turn at the previous turn's end rounded down to 32 tokens
        # (prefill_producer.send_chunk), but Gemma4PrefillRuntime only accepts chunk-aligned starts, so a
        # continuation fails validate_chunk. Off until the producer or the runtime supports unaligned continuations.
        "multi_turn_prob": 0,
        "seed": _env_int("GEMMA4_STRESS_SEED", 1234),
        "check_pcc": False,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


@contextmanager
def _producer(log_path, manifest_path):
    with log_path.open("w") as log:
        producer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "models.demos.common.prefill.runners.prefill_producer",
                "--manifest",
                str(manifest_path),
            ],
            env={**os.environ, "PREFILL_SEND_SHUTDOWN": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            yield
            assert producer.wait(timeout=300) == 0, log_path.read_text()[-4000:]
        finally:
            if producer.poll() is None:
                producer.terminate()
                producer.wait(timeout=30)


@pytest.mark.timeout(14400)
def test_prefill_serving_stress(monkeypatch, tmp_path):
    chunk_size = _env_int("GEMMA4_STRESS_CHUNK_SIZE", 8192)
    check_every = _env_int("GEMMA4_STRESS_CHECK_EVERY", 16)
    max_drift = float(os.environ.get("GEMMA4_STRESS_MAX_DRIFT", "1.25"))
    service_id = f"gemma4_stress_{os.getpid()}"
    monkeypatch.setenv("PREFILL_MANIFEST", str(Path(__file__).parents[1] / "tt/runners/manifest.json"))
    monkeypatch.setenv("PREFILL_H2D_SERVICE_ID", service_id)
    monkeypatch.setenv("PREFILL_NUM_USERS", str(_NUM_USERS))
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", str(chunk_size))
    monkeypatch.setattr(Gemma4ServiceConfig, "CHUNK_SIZE", chunk_size)
    from models.demos.common.prefill.runners import prefill_runner

    chunk_seconds = []
    chunk_depths = []
    original_prefill = Gemma4PrefillRuntime.prefill_chunk
    original_loop = prefill_runner.run_request_loop

    def timed_prefill(runtime, input_tensor, kv_cache, **request):
        start = time.perf_counter()
        original_prefill(runtime, input_tensor, kv_cache, **request)
        chunk_seconds.append(time.perf_counter() - start)
        chunk_depths.append(request["actual_start"] // _DEPTH_BUCKET_TOKENS)
        if len(chunk_seconds) % check_every == 0:
            hidden = ttnn.to_torch(ttnn.get_device_tensors(runtime.output)[0]).float()
            assert torch.isfinite(hidden).all(), f"non-finite output after {len(chunk_seconds)} chunks"
            assert hidden.std() > 1e-3, f"collapsed output after {len(chunk_seconds)} chunks"

    def checked_loop(runtime, kv_cache, *args, **kwargs):
        channel = ttnn.InterProcessCounterChannel.connect(
            f"/tt_prefill_layer_acks_{service_id}", connect_timeout_ms=30000
        )
        acks = 0
        original_loop(runtime, kv_cache, *args, **kwargs)
        expected = len(chunk_seconds) * runtime.config.num_layers
        deadline = time.monotonic() + 60
        while acks < expected and time.monotonic() < deadline:
            acks += channel.try_consume_all()
            time.sleep(0.01)
        assert acks == expected, f"{acks} layer acks for {len(chunk_seconds)} chunks"

    monkeypatch.setattr(Gemma4PrefillRuntime, "prefill_chunk", timed_prefill)
    monkeypatch.setattr(prefill_runner, "run_request_loop", checked_loop)
    manifest = _write_producer_manifest(tmp_path / "producer.json", chunk_size)
    with _producer(tmp_path / "producer.log", manifest):
        prefill_runner.main()

    assert chunk_seconds, "no chunks were served"
    half = len(chunk_seconds) // 2
    drift = {}
    for bucket in sorted(set(chunk_depths)):
        early = [t for i, (t, d) in enumerate(zip(chunk_seconds, chunk_depths)) if d == bucket and 8 <= i < half]
        late = [t for i, (t, d) in enumerate(zip(chunk_seconds, chunk_depths)) if d == bucket and i >= half]
        if len(early) >= 4 and len(late) >= 4:
            drift[bucket] = statistics.median(late) / statistics.median(early)
    logger.info(
        f"stress: {len(chunk_seconds)} chunks of {chunk_size} in "
        f"{sum(chunk_seconds):.1f} s; chunk latency median {statistics.median(chunk_seconds) * 1e3:.1f} ms, "
        f"max {max(chunk_seconds) * 1e3:.1f} ms (chunk {chunk_seconds.index(max(chunk_seconds))}); late/early by {_DEPTH_BUCKET_TOKENS // 1024}k depth bucket: "
        + ", ".join(f"{b * _DEPTH_BUCKET_TOKENS // 1024}k {r:.3f}" for b, r in drift.items())
    )
    slowest = sorted(range(len(chunk_seconds)), key=chunk_seconds.__getitem__, reverse=True)[:5]
    logger.info(
        "stress: slowest chunks (index, depth bucket, ms): "
        + ", ".join(f"({i}, {chunk_depths[i] * _DEPTH_BUCKET_TOKENS // 1024}k, {chunk_seconds[i] * 1e3:.1f})" for i in slowest)
    )
    assert drift, "too few chunks per depth bucket to compare early and late latency"
    worst = max(drift, key=drift.get)
    assert drift[worst] <= max_drift, f"chunk latency drifted {drift[worst]:.3f}x at {worst * 32}k context"
