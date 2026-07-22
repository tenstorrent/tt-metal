# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy-friendly prefill trace capture for ``ops_perf_results_*.csv`` generation.

Tracy records the full session (model load, trace capture, warm replay). Steady-state
inference is trace replay only; filter the CSV to ops between ``start`` and ``stop``
signposts on the measured replay.

Use ``test_prefill_trace_perf.py`` for capture-vs-replay amortization under plain pytest.

Example (31B blackhole 1×4):

    rm -rf generated/profiler/.logs generated/profiler/reports
    export HF_MODEL=google/gemma-4-31b-it
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

    python -m tracy -p -r -v -m pytest \\
        models/demos/gemma4/tests/unit/test_prefill_trace_tracy_csv.py \\
        -k batch1-prefill_128-1x4 -v -s --timeout=1800

    python tools/tracy/process_ops_logs.py --date
    # Filter ops_perf_results_*.csv to rows between signposts "start" and "stop"
"""

import pytest

from models.demos.gemma4.demo.text_demo import _batch_prefill_hits_ceiling, _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator_trace import (
    GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS,
    GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
    GEMMA4_TRACE_PREFILL_SEQ_LENS,
    can_gemma4_enable_prefill_trace,
)
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from ..test_factory import TestFactory, _get_model_path, parametrize_mesh_with_fabric
from .test_prefill_trace_parity import _PREFILL_TRACE_BATCH_SIZES, _PREFILL_TRACE_BUCKETS
from .tracy_prefill_common import run_prefill_trace_tracy_session


@pytest.mark.gemma4_prefill_trace
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _PREFILL_TRACE_BUCKETS, ids=lambda n: f"prefill_{n}")
@pytest.mark.parametrize("batch_size", _PREFILL_TRACE_BATCH_SIZES, ids=lambda b: f"batch{b}")
def test_prefill_trace_tracy_csv(batch_size, prefill_len, mesh_device, reset_seeds, request):
    """Full Tracy session: load, capture, warm replay, signposted measured replay."""
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

    run_prefill_trace_tracy_session(
        mesh_device,
        model_path,
        batch_size,
        prefill_len,
        hf_config.vocab_size,
        emit_signposts=True,
    )
