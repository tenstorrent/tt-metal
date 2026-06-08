# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill trace amortization: capture once, replay once.

Exercises a single (batch, ISL) bucket without the full batched trace warmup sweep.
For Tracy ``ops_perf_results_*.csv`` with ``start``/``stop`` signposts on replay, use
``test_prefill_trace_tracy_csv.py`` under ``python -m tracy``.

Example (31B blackhole 1×4):

    export HF_MODEL=google/gemma-4-31b-it
    pytest models/demos/gemma4/tests/unit/test_prefill_trace_perf.py \\
        -k batch1-prefill_128-1x4 -v -s --timeout=1800
"""

import os

import pytest
from loguru import logger

from models.demos.gemma4.demo.text_demo import _batch_prefill_hits_ceiling, _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator_trace import (
    GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS,
    GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
    GEMMA4_TRACE_PREFILL_SEQ_LENS,
    can_gemma4_enable_prefill_trace,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from ..test_factory import TestFactory, _get_model_path, parametrize_mesh_with_fabric
from .test_prefill_trace_parity import _PREFILL_TRACE_BATCH_SIZES, _PREFILL_TRACE_BUCKETS
from .tracy_prefill_common import (
    build_prefill_trace_fixtures,
    load_prefill_trace_generator,
    run_prefill_trace_capture,
    run_prefill_trace_replay,
)


@pytest.mark.gemma4_prefill_trace
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _PREFILL_TRACE_BUCKETS, ids=lambda n: f"prefill_{n}")
@pytest.mark.parametrize("batch_size", _PREFILL_TRACE_BATCH_SIZES, ids=lambda b: f"batch{b}")
def test_prefill_trace_perf(batch_size, prefill_len, mesh_device, reset_seeds, request):
    """Capture prefill device trace for one bucket, then replay once (amortization).

    Skips the all-batch × all-ISL trace warmup sweep (``already_warmed_up_prefill``).
    """
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models disable prefill trace")

    kernel_len = get_padded_prefill_len(prefill_len)
    if _batch_prefill_hits_ceiling(batch_size, prefill_len):
        pytest.skip(
            f"batch {batch_size} x kernel {kernel_len} meets 128k batched-prefill ceiling "
            f"(trace warmup skips this combo)"
        )

    if kernel_len not in GEMMA4_TRACE_PREFILL_SEQ_LENS:
        pytest.skip(f"kernel_len={kernel_len} not in trace ISL buckets {GEMMA4_TRACE_PREFILL_SEQ_LENS}")

    max_padded_batch = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size)
    if not can_gemma4_enable_prefill_trace(kernel_len, batch_size=max_padded_batch):
        pytest.skip(
            f"prefill trace disabled for padded_batch={max_padded_batch} x kernel={kernel_len} "
            f"(ISL>{GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN} or "
            f"batch×kernel>={GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS})"
        )

    model_path = _get_model_path()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    fixtures = build_prefill_trace_fixtures(batch_size, prefill_len, hf_config.vocab_size)
    mesh_key = "x".join(str(d) for d in mesh_device.shape)
    logger.info(
        "Prefill trace perf: model={} mesh={} batch={} prompt_len={} kernel_len={}",
        os.path.basename(model_path.rstrip("/")),
        mesh_key,
        batch_size,
        prefill_len,
        fixtures["kernel_len"],
    )

    generator, kv_caches = load_prefill_trace_generator(mesh_device, model_path, fixtures)
    profiler = BenchmarkProfiler()

    profiler.start("trace_capture")
    run_prefill_trace_capture(generator, kv_caches, fixtures)
    profiler.end("trace_capture")

    profiler.start("trace_replay")
    run_prefill_trace_replay(generator, kv_caches, fixtures)
    profiler.end("trace_replay")

    capture_s = profiler.get_duration("trace_capture")
    replay_s = profiler.get_duration("trace_replay")
    speedup = capture_s / replay_s if replay_s > 0 else float("inf")

    logger.info(
        "Prefill trace perf timings: capture={:.3f}s replay={:.3f}s speedup={:.2f}x",
        capture_s,
        replay_s,
        speedup,
    )

    assert (
        replay_s < capture_s
    ), f"Trace replay ({replay_s:.3f}s) should be faster than capture+compile ({capture_s:.3f}s)"
