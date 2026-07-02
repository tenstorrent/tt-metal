# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Chunked-prefill device-perf vs e2e-perf driver (Kimi K2.6, 5 layers, 1 chunk = 0 KV cache, 5120 tokens).

A single combined driver that quantifies the host/dispatch "tax" — the gap between the device's pure
compute time and the actual end-to-end wall-clock per chunk (see tt/runners/H2D_DISPATCH_TAX.md and
prior op2op work). It drives the no-PCC worker
(test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc) twice:

  1. compile run once + device-perf (tracy): the worker JIT-compiles in a warmup chunk bracketed by
     PROFILE_WARMUP_START / PROFILE_MEASURE_START, then runs ONE measured chunk closed by
     PROFILE_MEASURE_END. We sum the merged multi-device DEVICE KERNEL DURATION between
     (PROFILE_MEASURE_START, PROFILE_MEASURE_END) → device-kernel ns for 1 chunk of 5 layers.
  2. compile run once + 10 standalone loops (no tracy): the worker writes its per-iter wall-clock to
     TT_PREFILL_PERF_JSON; we average them → e2e ns per chunk (1 chunk/iter).

It then prints device perf, e2e perf, and the loss (e2e − device kernel = host/dispatch overhead),
in ms and as a percentage of e2e.

Like the other drivers in this dir (test_moe_perf.py, test_prefill_block_perf.py) this test takes NO
device fixtures: each phase opens/closes the mesh in its own subprocess, so the driver holds no chip
locks. Requires an 8x4 Blackhole mesh + the Kimi TTNN weight cache (TT_KIMI_PREFILL_TTNN_CACHE +
KIMI_K2_6_HF_MODEL); without them the worker subprocess skips and this test fails on the missing CSV/
JSON — run it only where the worker can run.
"""

import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.demos.deepseek_v3_d_p.utils.perf_utils import measure_device_perf_ns, run_e2e_wall_clock
from models.demos.deepseek_v3_d_p.utils.perlayer_op2op import perlayer_report

_WORKER = (
    "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py"
    "::test_kimi_prefill_transformer_chunked_no_pcc"
)
_SUBDIR = "deepseek_v3_prefill_chunked"

# `-k` is substring-matched: "L5" is unambiguous against the other layer ids (L1/L10/L61).
# "iters1"/"ten_iters" are unambiguous against the other parametrize ids. The worker always runs a
# single chunk, so there is no chunk id to disambiguate. "no_trace" picks the plain per-op dispatch
# baseline (this driver measures the op2op tax; the trace variant collapses it and is run separately).
_SELECT_DEVICE = "L5 and iters1 and no_trace"
_SELECT_E2E = "L5 and ten_iters and no_trace"

# TT_PREFILL_PROFILE_WARMUP=1 makes the worker run one compile/warmup chunk (excluded via signposts)
# before the measured region — passed via extra_env, NEVER prefixed into the command (tracy's -m
# mis-parses leading KEY=VAL tokens).
_WARMUP_ENV = {"TT_PREFILL_PROFILE_WARMUP": "1"}


@pytest.mark.timeout(0)
def test_kimi_prefill_chunked_e2e_vs_device_l5_chunks1():
    # --- Phase 1: device perf on ONE WARM chunk. The worker runs a compile/warmup chunk first (kernels
    # JITted, then the on-device profiler buffer is flushed) and the measured chunk is profiled warm, so
    # op-to-op gaps reflect steady-state host dispatch (not JIT). PROFILE_MEASURE_START/END bracket the
    # warm forward; we both sum device-kernel ns and emit the per-layer / per-op + op2op breakdown. ---
    logger.info("=== Phase 1: device perf (tracy, 1 warm chunk; compile run excluded) ===")
    device_ns, _, _ = measure_device_perf_ns(
        command=f"pytest {_WORKER} -k '{_SELECT_DEVICE}'",
        subdir=_SUBDIR,
        between_signposts=("PROFILE_MEASURE_START", "PROFILE_MEASURE_END"),
        extra_env=_WARMUP_ENV,
    )

    # Per-layer, per-op device-kernel + op2op-gap breakdown from the same warm CSV (worst/critical-path
    # device). The CSV survives Phase 2 (e2e runs without tracy, so it doesn't clear generated/profiler/).
    device_csv = get_latest_ops_log_filename(_SUBDIR)
    logger.info("=== Per-layer per-op device-kernel vs op2op (warm chunk) ===\n" + perlayer_report(device_csv))

    # --- Phase 2: e2e wall-clock averaged over 10 standalone loops (no tracy, 1 chunk/iter). Warm:
    # the worker runs a compile chunk first (excluded), then times the 10 measured loops. ---
    logger.info("=== Phase 2: e2e perf (10 standalone loops, wall-clock) ===")
    e2e = run_e2e_wall_clock(command=f"pytest {_WORKER} -k '{_SELECT_E2E}'", extra_env=_WARMUP_ENV)
    e2e_ns = e2e["avg_iter_seconds"] * 1e9  # 1 chunk per iter → per-iter == per-chunk

    # --- Report: the loss is the host/dispatch/sync overhead not covered by device compute. ---
    overhead_ns = e2e_ns - device_ns
    overhead_pct = (overhead_ns / e2e_ns * 100) if e2e_ns else float("nan")

    logger.info("=" * 72)
    logger.info("Kimi K2.6 chunked prefill — 5 layers, 1 chunk (0 KV cache, 5120 tokens)")
    logger.info("-" * 72)
    logger.info(f"  Device perf (kernel) : {device_ns / 1e6:10.3f} ms  ({device_ns / 1e3:12,.1f} us)")
    logger.info(f"  E2E perf (wall-clock): {e2e_ns / 1e6:10.3f} ms  ({e2e_ns / 1e3:12,.1f} us)")
    logger.info(f"  Loss (host/dispatch) : {overhead_ns / 1e6:10.3f} ms  ({overhead_pct:6.1f}% of e2e)")
    logger.info(f"  e2e loops averaged   : {e2e['num_iters']}  per-iter (s): {e2e['per_iter_seconds']}")
    logger.info("=" * 72)

    # Sanity: device kernel time must be a positive fraction of the e2e wall-clock (the loss is the
    # whole point of this test, so we don't gate on a tight baseline — record the number first).
    assert device_ns > 0, "no device-kernel time measured between PROFILE_MEASURE_START/END"
    assert e2e_ns > 0, "no e2e wall-clock measured"
    assert device_ns <= e2e_ns * 1.05, (
        f"device kernel ns ({device_ns:,.0f}) exceeds e2e ns ({e2e_ns:,.0f}) — "
        "measurement mismatch (different chunk/layer count between phases?)"
    )
