# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Perf comparison for the ring-fused indexer_score (Step E profiling; see ring_indexer_score_fusion_design.md).

Measures the wall-clock gain of the FUSED op (one program: all-gather co-scheduled with scoring, overlapped)
vs the UNFUSED baseline (two sequential ops: ring_attention all-gather THEN indexer_score_dsa) on the LoudBox
ring of 4.

Method: host wall-clock of a single COLD blocking dispatch, min over N iters (min = least host noise, most
device-dominated). Both variants do exactly ONE ttnn.synchronize_device per iter, so dispatch overhead is
comparable; the fused variant is one op vs the baseline's two. FRESH ccl semaphores every iteration -- the
all-gather signals into GLOBAL monotonic semaphores, so reusing them would let the fused consumer's wait_min
pass instantly (gating vanishes) and understate the baseline / misrepresent the fused cost. Inputs/buffers are
built once and reused; only the semaphores are recreated per iter.

Reported delta = (AG + score) - fused = the AG transport the fusion hid behind compute (plus one fewer op
boundary). Includes a small fixed host overhead (roughly equal for both), so read the trend, not the absolute.

Run:  INDEXER_FUSED_PERF=1 scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score_lb_ring4_perf.py
"""

import os
import time
import statistics

import pytest
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score import (
    glx_config,
    QB_CASES,
    QB_IDS,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.test_indexer_score_lb_ring4_ag_equiv import (
    _open_ring4_ccl,
    _close_ring4_ccl,
    _build_ring4_fused_inputs,
    _ring_attention_ag,
    _reset,
    SP_AXIS,
)

pytestmark = [
    pytest.mark.skipif(os.environ.get("INDEXER_FUSED_PERF") != "1", reason="set INDEXER_FUSED_PERF=1 to run"),
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ring-of-4 needs the 8-chip LoudBox (2x4)"),
]

_ITERS = 20


def _time_min_us(fn, submesh, stall_group):
    """Warm up once (compile), then time N cold blocking dispatches; return min microseconds."""
    fn()
    ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
    samples = []
    for _ in range(_ITERS):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        samples.append((time.perf_counter() - t0) * 1e6)
    return min(samples), statistics.median(samples)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_ring4_fused_vs_unfused_perf(case_id, heads, block_cyclic):
    """Host wall-clock of FUSED (one overlapped program) vs UNFUSED (all-gather then score) on the ring of 4.
    Informational/logged: host time is dominated by the fixed per-dispatch fabric-setup overhead, so the
    device-level overlap signal lives in test_indexer_score_lb_ring4_profile.py, not in a hard assert here."""
    submesh, parent, _unused_sems, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_dev, w_dev, k_local, k_gathered, bc, sems = _build_ring4_fused_inputs(submesh, heads, block_cyclic)

        def fused_once():
            _reset(sems)
            ttnn.experimental.indexer_score_dsa_fused(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                sems,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                program_config=glx_config(heads),
                **bc,
            )

        def unfused_once():
            _reset(sems)
            k_full = _ring_attention_ag(k_local, k_gathered, sems, subdevice_id)
            ttnn.experimental.indexer_score_dsa(
                q_dev, k_full, w_dev, cluster_axis=SP_AXIS, program_config=glx_config(heads), **bc
            )

        def ag_only():
            _reset(sems)
            _ring_attention_ag(k_local, k_gathered, sems, subdevice_id)

        def score_only():
            # scores the full-grid indexer over the gathered buffer (data irrelevant to timing).
            ttnn.experimental.indexer_score_dsa(
                q_dev, k_gathered, w_dev, cluster_axis=SP_AXIS, program_config=glx_config(heads), **bc
            )

        ag_min, _ = _time_min_us(ag_only, submesh, stall_group)
        score_min, _ = _time_min_us(score_only, submesh, stall_group)
        fused_min, _ = _time_min_us(fused_once, submesh, stall_group)
        unfused_min, _ = _time_min_us(unfused_once, submesh, stall_group)

        layout = "block_cyclic" if block_cyclic else "contiguous"
        gain = unfused_min - fused_min
        # Overlap ceiling: perfect fusion approaches max(AG, score); the floor it must beat is AG+score.
        logger.info(
            f"[{case_id}-{layout}] min us: AG-only={ag_min:.1f} score-only={score_min:.1f} "
            f"| UNFUSED(AG+score)={unfused_min:.1f} FUSED={fused_min:.1f} ideal~max={max(ag_min,score_min):.1f} "
            f"| gain={gain:.1f}us ({100*gain/unfused_min:.1f}%) speedup={unfused_min/fused_min:.2f}x"
        )
        # Sanity only -- both dispatches produced a timing. This is a logged perf probe, not a pass/fail gate
        # (host wall-clock does not reflect the device-level overlap; see the profiler test).
        assert fused_min > 0 and unfused_min > 0
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)
