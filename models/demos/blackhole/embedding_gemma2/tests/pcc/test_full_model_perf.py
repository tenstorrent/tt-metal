# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warmed, traced end-to-end latency of the full model (inputs on device -> normalized embedding).

test_full_perf_traced   one trace per case (captured together), warmed, then:
                        `device` = N back-to-back execute_trace + one synchronize, per forward
                                   (persistent inputs already resident; towers + merge + text + pool)
                        `e2e`    = host->device input copy + replay + embedding readback, per request
                                   (excludes the HF processor / tokenizer on the host)
test_tracy_full_forward one warmed eager forward of one image item (+ short caption) between Tracy
                        signposts PERF_FULL / PERF_FULL_END (run under `python -m tracy`)

    pytest models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model_perf.py -k traced -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import log_result, make_batch
from models.demos.blackhole.embedding_gemma2.tests.pcc.test_full_model import full_items, full_model  # noqa: F401
from models.demos.blackhole.embedding_gemma2.tt.embedding_model import ProcessedItem

try:
    from tracy import signpost
except ImportError:  # tracy is optional outside profiling runs

    def signpost(*_args, **_kwargs):
        pass


# case -> (catalogue item, batch) or ("text", batch, tokens)
PERF_CASES = {
    "text_b1_s128": ("text", 1, 128),
    "text_b1_s512": ("text", 1, 512),
    "text_b8_s512": ("text", 8, 512),
    "image_caption": ("image_dog_caption", 1),
    "audio10s_text": ("audio_speech10s_text", 1),
    "video8": ("video_text", 1),
}
N_WARMUP = 3
N_ITERS = int(os.environ.get("EMBEDDING_GEMMA2_PERF_ITERS", "20"))


def _processed(m, case, items, snapshot) -> list[ProcessedItem]:
    spec = PERF_CASES[case]
    if spec[0] == "text":
        _, batch, n = spec
        ids, mask = make_batch(snapshot, [n] * batch)
        return [
            ProcessedItem({"input_ids": ids[b : b + 1], "attention_mask": mask[b : b + 1]}, ids[b], 0, 0, 0, 0)
            for b in range(batch)
        ]
    return [m.process(items[spec[0]])] * spec[1]


@pytest.fixture(scope="module")
def perf_hosts(full_model, full_items, eg2_snapshot):
    m = full_model
    hosts = {c: m.prepare_batch(_processed(m, c, full_items, eg2_snapshot)) for c in PERF_CASES}
    m.capture_all([h.key for h in hosts.values()])
    yield hosts
    m.release_traces()


@pytest.mark.parametrize("case", list(PERF_CASES))
def test_full_perf_traced(case, full_model, perf_hosts):
    m, host = full_model, perf_hosts[case]
    key = host.key
    out = m.replay(host)
    for _ in range(N_WARMUP):
        m.execute_captured(key)
    ttnn.synchronize_device(m.device)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        m.execute_captured(key)
    ttnn.synchronize_device(m.device)
    device_ms = (time.perf_counter() - t0) * 1e3 / N_ITERS

    e2e = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        out = m.read_embedding(m.replay(host), key.batch)
        e2e.append((time.perf_counter() - t0) * 1e3)
    e2e_ms = sum(e2e) / len(e2e)
    assert torch.isfinite(out).all() and torch.allclose(out.norm(dim=-1), torch.ones(key.batch), atol=1e-3)
    logger.info(f"PERF {case} {key}: device {device_ms:.3f} ms/forward; e2e {e2e_ms:.3f} ms (min {min(e2e):.3f})")
    log_result("full_latency_device_ms", device_ms, 0.0, case=case, key=str(key), iters=N_ITERS)
    log_result("full_latency_e2e_ms", e2e_ms, 0.0, case=case, key=str(key), iters=N_ITERS)


def test_tracy_full_forward(eg2_device, full_model, full_items):
    m = full_model
    host = m.prepare_batch([m.process(full_items["image_dog_caption"])])
    x = m.to_device(host)
    m.prepare(host.key, x)
    ttnn.deallocate(m.forward(x, host.key))  # compile
    ttnn.synchronize_device(eg2_device)
    ttnn.ReadDeviceProfiler(eg2_device)  # flush compile-run records
    signpost("PERF_FULL")
    out = m.forward(x, host.key)
    ttnn.synchronize_device(eg2_device)
    signpost("PERF_FULL_END")
    ttnn.ReadDeviceProfiler(eg2_device)
    ttnn.deallocate(out)
