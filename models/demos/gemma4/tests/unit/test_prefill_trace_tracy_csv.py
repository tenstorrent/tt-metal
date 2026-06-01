# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-pass traced prefill for Tracy ``ops_perf_results_*.csv`` generation (#44957).

Model load is kept out of the Tracy capture window: host op logging is suppressed during
``from_pretrained``, device profiler buffers are flushed, then one traced prefill capture
is recorded (with optional partial Tracy profiling via ``-p``).

Use ``test_prefill_trace_perf.py`` for capture-vs-replay amortization under plain pytest.

Example (31B on QB2 1×4):

    rm -rf generated/profiler/.logs generated/profiler/reports
    export HF_MODEL=google/gemma-4-31b-it
    export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

    python -m tracy -r -p -v -m pytest \\
        models/demos/gemma4/tests/unit/test_prefill_trace_tracy_csv.py \\
        -k batch1-prefill_128-1x4 -v -s --timeout=1800

    python tools/tracy/process_ops_logs.py --date
    ls generated/profiler/reports/*/ops_perf_results_*.csv
"""

import os

import pytest
from loguru import logger

from models.demos.gemma4.demo.text_demo import _batch_prefill_hits_ceiling, _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import GEMMA4_TRACE_PREFILL_SEQ_LENS
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
from .test_prefill_trace_perf import (
    _flush_device_profiler,
    _maybe_signpost,
    _suppress_ttnn_op_profiler,
    _traced_prefill,
)


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _PREFILL_TRACE_BUCKETS, ids=lambda n: f"prefill_{n}")
@pytest.mark.parametrize("batch_size", _PREFILL_TRACE_BATCH_SIZES, ids=lambda b: f"batch{b}")
def test_prefill_trace_tracy_csv(batch_size, prefill_len, mesh_device, reset_seeds, request):
    """Run one traced prefill capture inside the Tracy profile window."""
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
        "Prefill trace Tracy CSV: model={} mesh={} batch={} prompt_len={} kernel_len={}",
        os.path.basename(model_path.rstrip("/")),
        mesh_key,
        batch_size,
        prefill_len,
        kernel_len,
    )

    max_batch_size = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size)
    with _suppress_ttnn_op_profiler():
        generator, kv_caches, _tokenizer = Gemma4Generator.from_pretrained(
            mesh_device=mesh_device,
            model_path=model_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_cfg,
        )

    _flush_device_profiler(mesh_device)

    model_args = generator.model_args[0]
    assert model_args.can_enable_trace(kernel_len), f"Trace not enabled for kernel_len={kernel_len}"
    generator.already_warmed_up_prefill = True

    tag = f"batch{batch_size}_prefill{prefill_len}_kernel{kernel_len}"

    tracy_profiler = None
    try:
        from tracy import Profiler

        tracy_profiler = Profiler()
        tracy_profiler.enable()
    except ImportError:
        pass

    try:
        _maybe_signpost("prefill_trace_capture", tag)
        _traced_prefill(generator, kv_caches, tokens, page_table, prompt_lens)
    finally:
        if tracy_profiler is not None:
            tracy_profiler.disable()

    _flush_device_profiler(mesh_device)
