# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warmed, traced latency of the audio forward (subsampling -> 12 conformer layers -> output_proj -> embed_audio).

test_audio_perf_traced      one trace per (batch, bucket), warmed, then N timed replays.
                            `device` = execute_trace + synchronize (inputs already resident)
                            `e2e`    = host->device input copy + replay + soft-token readback
                            `host_prep` = AudioEncoder.prepare_inputs (padding, masks)
                            Feature extraction (log-mel) is the processor and is not timed here.
test_tracy_audio_forward    one warmed eager forward between Tracy signposts PERF_AUDIO / PERF_AUDIO_END
                            (run under `python -m tracy`, see doc/functional_audio/README.md)

Run:
    pytest models/demos/embedding_gemma2/tests/pcc/audio/test_audio_perf.py -k traced -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.embedding_gemma2.tests.common import log_result
from models.demos.embedding_gemma2.tests.pcc.audio.audio_common import device_inputs, load_case
from models.demos.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

try:
    from tracy import signpost
except ImportError:  # tracy is optional outside profiling runs

    def signpost(*_args, **_kwargs):
        pass


# (batch, token bucket, clip names concatenated along the batch)
PERF_CASES = (
    (1, 256, ("s10",)),
    (1, 768, ("s30",)),
    (4, 256, ("s10", "s10", "s10", "s10")),
)
N_WARMUP = 3
N_ITERS = int(os.environ.get("EMBEDDING_GEMMA2_PERF_ITERS", "20"))


def _inputs(snapshot, cases):
    feats, masks = zip(*(load_case(str(snapshot), c) for c in cases))
    return torch.cat(feats), torch.cat(masks)


@pytest.fixture(scope="module")
def perf_encoder(eg2_device, tt_audio_model, eg2_snapshot):
    enc = AudioEncoder(tt_audio_model, eg2_device, eg2_snapshot)
    enc.capture_all([(b, s) for b, s, _ in PERF_CASES])
    yield enc
    enc.release_traces()


@pytest.mark.parametrize("batch,bucket,cases", PERF_CASES, ids=[f"b{b}_s{s}" for b, s, _ in PERF_CASES])
def test_audio_perf_traced(batch, bucket, cases, perf_encoder, eg2_snapshot):
    time_traced(perf_encoder, batch, bucket, cases, eg2_snapshot, "bf16")


def time_traced(perf_encoder, batch, bucket, cases, eg2_snapshot, act):
    feats, mask = _inputs(eg2_snapshot, cases)
    t0 = time.perf_counter()
    host = perf_encoder.prepare_inputs(feats, mask)
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
        out = AudioEncoder.read_rows(soft, host.num_tokens)
        e2e.append((time.perf_counter() - t0) * 1e3)
    e2e_ms = sum(e2e) / len(e2e)
    assert all(torch.isfinite(o).all() for o in out)

    seconds = float(mask.sum(-1).max()) / 100.0
    logger.info(
        f"PERF audio ({act} act) b{batch} s{bucket} ({seconds:.1f} s clips): device {device_ms:.3f} ms/forward "
        f"({batch / device_ms * 1e3:.1f} clips/s); e2e {e2e_ms:.3f} ms (min {min(e2e):.3f}); host prep {host_ms:.1f} ms"
    )
    fields = dict(batch=batch, bucket=bucket, act=act)
    log_result("audio_latency_device_ms", device_ms, 0.0, iters=N_ITERS, **fields)
    log_result("audio_latency_e2e_ms", e2e_ms, 0.0, iters=N_ITERS, **fields)
    log_result("audio_host_prep_ms", host_ms, 0.0, **fields)


@pytest.mark.parametrize(
    "batch,bucket,cases", [PERF_CASES[1], PERF_CASES[2], PERF_CASES[0]], ids=["b1_s768", "b4_s256", "b1_s256"]
)
def test_tracy_audio_forward(batch, bucket, cases, eg2_device, tt_audio_model, eg2_snapshot):
    enc = AudioEncoder(tt_audio_model, eg2_device, eg2_snapshot)
    _, x = device_inputs(enc, *_inputs(eg2_snapshot, cases))
    enc.forward(x)  # compile
    ttnn.synchronize_device(eg2_device)
    ttnn.ReadDeviceProfiler(eg2_device)  # flush compile-run records
    signpost("PERF_AUDIO")
    enc.forward(x)
    ttnn.synchronize_device(eg2_device)
    signpost("PERF_AUDIO_END")
    ttnn.ReadDeviceProfiler(eg2_device)
