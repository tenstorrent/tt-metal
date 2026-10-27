# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warmed, traced latency of the text embedding forward.

test_perf_traced      captures one trace per (batch, bucket), warms it, then times N replays.
                      `device` = execute_trace + synchronize (inputs already resident)
                      `e2e`    = host->device input copy + replay + embedding readback
test_tracy_forward    one warmed eager forward between Tracy signposts PERF_TEXT / PERF_TEXT_END
                      (run under `python -m tracy`, see doc/functional_text/README.md)

Run:
    pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py -k traced -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import log_result, make_batch
from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder

try:
    from tracy import signpost
except ImportError:  # tracy is optional outside profiling runs

    def signpost(*_args, **_kwargs):
        pass


PERF_CASES = ((1, 128), (1, 512), (1, 2048), (8, 512))
N_WARMUP = 3
N_ITERS = int(os.environ.get("EMBEDDING_GEMMA2_PERF_ITERS", "20"))


@pytest.fixture(scope="module")
def encoder(eg2_device, tt_text_model):
    enc = TextEncoder(tt_text_model, eg2_device)
    enc.capture_all(PERF_CASES)
    yield enc
    enc.release_traces()


@pytest.mark.parametrize("batch,seq_len", PERF_CASES, ids=[f"b{b}_s{s}" for b, s in PERF_CASES])
def test_perf_traced(batch, seq_len, encoder, eg2_snapshot):
    ids, mask = make_batch(eg2_snapshot, [seq_len] * batch)
    host = encoder.prepare_inputs(ids, mask)
    assert host.seq_len == seq_len
    dev = encoder.device

    last, emb = encoder.replay(host)
    for _ in range(N_WARMUP):
        encoder.execute_captured(batch, seq_len)
    ttnn.synchronize_device(dev)

    # device time: back-to-back replays, one sync at the end
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        encoder.execute_captured(batch, seq_len)
    ttnn.synchronize_device(dev)
    device_ms = (time.perf_counter() - t0) * 1e3 / N_ITERS

    # end to end: upload + replay + embedding readback, per request
    e2e = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        _, emb = encoder.replay(host)
        out = TextEncoder.read_embedding(emb, batch)
        e2e.append((time.perf_counter() - t0) * 1e3)
    e2e_ms = sum(e2e) / len(e2e)
    assert torch.isfinite(out).all()

    tokens = batch * seq_len
    logger.info(
        f"PERF b{batch} s{seq_len}: device {device_ms:.3f} ms/forward ({tokens / device_ms * 1e3:.0f} tok/s, "
        f"{batch / device_ms * 1e3:.1f} emb/s); e2e {e2e_ms:.3f} ms (min {min(e2e):.3f})"
    )
    log_result("latency_device_ms", device_ms, 0.0, batch=batch, seq_len=seq_len, iters=N_ITERS)
    log_result("latency_e2e_ms", e2e_ms, 0.0, batch=batch, seq_len=seq_len, iters=N_ITERS)


@pytest.mark.parametrize("batch,seq_len", [(1, 512), (8, 512)], ids=["b1_s512", "b8_s512"])
def test_tracy_forward(batch, seq_len, eg2_device, tt_text_model, eg2_snapshot):
    enc = TextEncoder(tt_text_model, eg2_device)
    ids, mask = make_batch(eg2_snapshot, [seq_len] * batch)
    host = enc.prepare_inputs(ids, mask)
    enc.model.prepare(host.seq_len)
    ids_d, keep_d = enc._to_device(host)
    enc.forward(ids_d, keep_d)  # compile
    ttnn.synchronize_device(eg2_device)
    ttnn.ReadDeviceProfiler(eg2_device)  # flush compile-run records
    signpost("PERF_TEXT")
    enc.forward(ids_d, keep_d)
    ttnn.synchronize_device(eg2_device)
    signpost("PERF_TEXT_END")
    ttnn.ReadDeviceProfiler(eg2_device)
