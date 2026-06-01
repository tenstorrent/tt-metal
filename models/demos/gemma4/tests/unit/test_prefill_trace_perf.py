# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill trace amortization: capture once, replay once (#44957).

Exercises a single (batch, ISL) bucket without the full batched trace warmup sweep.
For Tracy ``ops_perf_results_*.csv``, use ``test_prefill_trace_tracy_csv.py`` instead
(device trace profiling supports only one trace execution per profiled run).

Example (31B on QB2 1×4):

    export HF_MODEL=google/gemma-4-31b-it
    pytest models/demos/gemma4/tests/unit/test_prefill_trace_perf.py \\
        -k batch1-prefill_128-1x4 -v -s --timeout=1800
"""

import os
from contextlib import contextmanager

import pytest
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo import _batch_prefill_hits_ceiling, _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import GEMMA4_TRACE_PREFILL_SEQ_LENS
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from ..test_factory import TestFactory, _get_model_path, parametrize_mesh_with_fabric
from .test_prefill_trace_parity import (
    _PREFILL_TRACE_BATCH_SIZES,
    _PREFILL_TRACE_BUCKETS,
    _build_tokens,
    _create_page_table,
    _page_params,
)

try:
    from tracy import signpost

    _HAS_SIGNPOST = True
except ModuleNotFoundError:
    _HAS_SIGNPOST = False


def _maybe_signpost(header: str, message: str = ""):
    if _HAS_SIGNPOST:
        signpost(header=header, message=message)


def _sync_mesh(mesh_device):
    ttnn.synchronize_device(mesh_device)


@contextmanager
def _suppress_ttnn_op_profiler():
    """Skip host-side TTNN op Tracy logging (``TTNN_OP_PROFILER`` is read per op)."""
    saved = os.environ.get("TTNN_OP_PROFILER")
    os.environ.pop("TTNN_OP_PROFILER", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["TTNN_OP_PROFILER"] = saved


def _flush_device_profiler(mesh_device):
    """Drain device profiler buffers so a later traced region does not overflow."""
    ttnn.ReadDeviceProfiler(mesh_device)
    _sync_mesh(mesh_device)


def _traced_prefill(generator, kv_cache, tokens, page_table, prompt_lens):
    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=True,
        warmup_prefill=False,
    )
    _sync_mesh(generator.model[0].mesh_device)
    return out


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

    model_path = _get_model_path()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    tokens, prompt_lens, kernel_len = _build_tokens(batch_size, prefill_len, hf_config.vocab_size)
    max_new_tokens = 32
    max_seq_len = max(prefill_len + max_new_tokens, 4096)
    paged_cfg = _page_params(batch_size, prefill_len, max_new_tokens)
    page_table = _create_page_table(batch_size, paged_cfg)

    mesh_key = "x".join(str(d) for d in mesh_device.shape)
    logger.info(
        "Prefill trace perf: model={} mesh={} batch={} prompt_len={} kernel_len={}",
        os.path.basename(model_path.rstrip("/")),
        mesh_key,
        batch_size,
        prefill_len,
        kernel_len,
    )

    max_batch_size = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size)
    generator, kv_caches, _tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_cfg,
    )
    model_args = generator.model_args[0]
    assert model_args.can_enable_trace(kernel_len), f"Trace not enabled for kernel_len={kernel_len}"

    # Skip the production batched trace warmup sweep; capture trace for this bucket only.
    generator.already_warmed_up_prefill = True

    tag = f"batch{batch_size}_prefill{prefill_len}_kernel{kernel_len}"
    profiler = BenchmarkProfiler()

    profiler.start("trace_capture")
    _maybe_signpost("prefill_trace_capture", tag)
    _traced_prefill(generator, kv_caches, tokens, page_table, prompt_lens)
    profiler.end("trace_capture")

    profiler.start("trace_replay")
    _maybe_signpost("prefill_trace_replay", tag)
    _traced_prefill(generator, kv_caches, tokens, page_table, prompt_lens)
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
