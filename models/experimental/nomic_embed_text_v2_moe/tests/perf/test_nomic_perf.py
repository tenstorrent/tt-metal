# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side performance baseline for the unoptimized port.

Nothing here asserts a target. The port has had no optimization pass, so there is no number to
regress against yet; these tests record the three latencies every later change is measured
from. The device-kernel half of the baseline is test_nomic_device_perf.py.

Deliberately measured with trace off, one command queue, DRAM-interleaved everything and the
bfloat16 / HiFi4 / fp32-accumulate config the correctness phase shipped. A baseline taken with
any of those already changed cannot attribute the change that follows.
"""

import time

import pytest
from loguru import logger

import ttnn

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.common import random_input_ids
from models.experimental.nomic_embed_text_v2_moe.reference.preprocessing import NomicPromptPrefix
from models.experimental.nomic_embed_text_v2_moe.tests.perf.perf_common import (
    BENCHMARK_SHAPES,
    BENCHMARK_TEXTS,
    HEADLINE_SHAPE,
    build_model,
    dram_allocated_bytes,
    measure,
    report,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model import encode

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MB = 1024 * 1024


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("batch, seqlen", BENCHMARK_SHAPES)
def test_model_latency(device, config, state_dict, batch, seqlen):
    """Backbone latency on an input already resident on device.

    Token ids are uploaded once outside the timed loop, so this isolates the model from
    host-to-device transfer. That is the number sharding, precision and program configs move;
    test_request_latency is the one trace and the host path move.
    """
    model, upload_s = build_model(device, config, state_dict)
    weights_mb = dram_allocated_bytes(device) / MB

    input_ids, attention_mask = random_input_ids(batch, seqlen, config)

    def run():
        ttnn.deallocate(model(input_ids, attention_mask))

    stats = measure(run, device)

    # Sampled after a forward has run and its output is freed, so this is residency between
    # calls rather than the transient peak inside the expert bank.
    resident_mb = dram_allocated_bytes(device) / MB
    report(
        logger,
        "backbone",
        batch,
        seqlen,
        stats,
        {
            "weights DRAM": f"{weights_mb:.0f} MB",
            "resident DRAM": f"{resident_mb:.0f} MB",
            "weight upload": f"{upload_s:.1f} s" if upload_s else "cached",
        },
    )


@pytest.mark.models_performance_bare_metal
def test_request_latency(device, config, state_dict, tokenizer):
    """One user request end to end: tokenize, upload, backbone, pool, normalize, read back.

    This is what a caller of `encode` waits for. It differs from test_model_latency by the host
    work the backbone measurement excludes on purpose, and the gap between the two is the budget
    for removing host boundaries and enabling trace.
    """
    model, _ = build_model(device, config, state_dict)

    def run():
        encode(model, tokenizer, BENCHMARK_TEXTS, prompt_prefix=NomicPromptPrefix.PASSAGE)

    stats = measure(run, device)

    tokenizer_start = time.perf_counter()
    encoded = tokenizer(list(BENCHMARK_TEXTS), padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokenize_ms = (time.perf_counter() - tokenizer_start) * 1000
    batch, seqlen = encoded["input_ids"].shape

    report(logger, "request (encode)", batch, seqlen, stats, {"tokenize": f"{tokenize_ms:.2f} ms"})


@pytest.mark.parametrize("batch, seqlen", [HEADLINE_SHAPE])
def test_device_perf_target(device, config, state_dict, batch, seqlen):
    """One warmed forward at the headline shape, bracketed by signposts for the profiler.

    run_device_perf sums every op in the log, so the measured region has to hold exactly one
    inference. The signposts exclude the warm-up forward, whose kernels are identical but whose
    program-cache misses would otherwise be attributed to the model.
    """
    from tracy import signpost

    model, _ = build_model(device, config, state_dict)
    input_ids, attention_mask = random_input_ids(batch, seqlen, config)

    ttnn.deallocate(model(input_ids, attention_mask))
    ttnn.synchronize_device(device)

    signpost("start")
    ttnn.deallocate(model(input_ids, attention_mask))
    ttnn.synchronize_device(device)
    signpost("stop")
