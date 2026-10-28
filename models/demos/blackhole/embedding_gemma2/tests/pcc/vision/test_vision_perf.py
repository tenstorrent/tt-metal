# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warmed, traced latency of the vision forward (patch embed -> 16 layers -> pool -> embed_vision).

test_vision_perf_traced     one trace per (batch, bucket), warmed, then N timed replays.
                            `device` = execute_trace + synchronize (inputs already resident)
                            `e2e`    = host->device input copy + replay + soft-token readback
                            `host_prep` = VisionEncoder.prepare_inputs (RoPE tables, pooling matrix, ...)
test_tracy_vision_forward   one warmed eager forward between Tracy signposts PERF_VISION / PERF_VISION_END
                            (run under `python -m tracy`, see doc/functional_vision/README.md)

Run:
    pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_perf.py -k traced -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import log_result
from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import load_case
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

try:
    from tracy import signpost
except ImportError:  # tracy is optional outside profiling runs

    def signpost(*_args, **_kwargs):
        pass


# (batch, bucket, cases concatenated along the batch)
PERF_CASES = (
    (1, 2560, ("landscape",)),
    (4, 2560, ("landscape", "portrait", "square", "small")),
    (8, 1280, ("video8",)),
)
N_WARMUP = 3
N_ITERS = int(os.environ.get("EMBEDDING_GEMMA2_PERF_ITERS", "20"))


def _inputs(snapshot, cases):
    pvs, pids = zip(*(load_case(str(snapshot), c) for c in cases))
    return torch.cat(pvs), torch.cat(pids)


@pytest.fixture(scope="module")
def perf_encoder(eg2_device, tt_vision_model):
    enc = VisionEncoder(tt_vision_model, eg2_device)
    enc.capture_all([(b, s) for b, s, _ in PERF_CASES])
    yield enc
    enc.release_traces()


@pytest.mark.parametrize("batch,bucket,cases", PERF_CASES, ids=[f"b{b}_s{s}" for b, s, _ in PERF_CASES])
def test_vision_perf_traced(batch, bucket, cases, perf_encoder, eg2_snapshot):
    pv, pid = _inputs(eg2_snapshot, cases)
    t0 = time.perf_counter()
    host = perf_encoder.prepare_inputs(pv, pid)
    host_ms = (time.perf_counter() - t0) * 1e3
    assert (host.batch, host.seq_len) == (batch, bucket)
    dev = perf_encoder.device

    perf_encoder.replay(host)
    for _ in range(N_WARMUP):
        perf_encoder.execute_captured(batch, bucket)
    ttnn.synchronize_device(dev)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        perf_encoder.execute_captured(batch, bucket)
    ttnn.synchronize_device(dev)
    device_ms = (time.perf_counter() - t0) * 1e3 / N_ITERS

    e2e = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        soft, _ = perf_encoder.replay(host)
        out = VisionEncoder.read_rows(soft, host.num_soft_tokens)
        e2e.append((time.perf_counter() - t0) * 1e3)
    e2e_ms = sum(e2e) / len(e2e)
    assert all(torch.isfinite(o).all() for o in out)

    logger.info(
        f"PERF vision b{batch} s{bucket}: device {device_ms:.3f} ms/forward ({batch / device_ms * 1e3:.1f} img/s); "
        f"e2e {e2e_ms:.3f} ms (min {min(e2e):.3f}); host prep {host_ms:.1f} ms"
    )
    log_result("vision_latency_device_ms", device_ms, 0.0, batch=batch, bucket=bucket, iters=N_ITERS)
    log_result("vision_latency_e2e_ms", e2e_ms, 0.0, batch=batch, bucket=bucket, iters=N_ITERS)
    log_result("vision_host_prep_ms", host_ms, 0.0, batch=batch, bucket=bucket)


@pytest.mark.parametrize("batch,bucket,cases", PERF_CASES[:2], ids=["b1_s2560", "b4_s2560"])
def test_tracy_vision_forward(batch, bucket, cases, eg2_device, tt_vision_model, eg2_snapshot):
    enc = VisionEncoder(tt_vision_model, eg2_device)
    x = enc.to_device(enc.prepare_inputs(*_inputs(eg2_snapshot, cases)))
    enc.forward(x)  # compile
    ttnn.synchronize_device(eg2_device)
    ttnn.ReadDeviceProfiler(eg2_device)  # flush compile-run records
    signpost("PERF_VISION")
    enc.forward(x)
    ttnn.synchronize_device(eg2_device)
    signpost("PERF_VISION_END")
    ttnn.ReadDeviceProfiler(eg2_device)
