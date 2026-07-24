# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance check for the production-scale sparse_sdpa_msa shapes.

Device timing via the real-time device program profiler (no tracy build needed): the op is profiled with
profile_realtime_program and its device kernel duration is asserted within +/- MSA_PERF_MARGIN of the
value measured on a Blackhole dev board. A symmetric band catches regressions AND unexpected speedups
(the latter often means work was silently skipped). Math utilization is not meaningful here (block-sparse
DMA + block-pool dominate, not the matmul), so we gate on wall-clock device duration instead.

Per-chip TP-shard shape: H=16, n_kv=1, S=640, T=56320 (nblk=440), topk=16, d=v_dim=128,
block_size=128.
Single-chip native GQA shape: H=64, n_kv=4, S=640, T=56320, topk=16, d=v_dim=128, block_size=128.
"""

import pytest
from loguru import logger

import ttnn

from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import make_msa_inputs, run_op_msa_native

# CI selects these by marker on the IOMMU-enabled BH sku (see ops_unit_tests.yaml); broad non-IOMMU jobs
# exclude the marker, and the perf helper fails (not skips) if the RT profiler is inactive, so an IOMMU
# regression on the pinned sku reds rather than passing silently.
pytestmark = [pytest.mark.requires_host_iommu, pytest.mark.use_module_device]

# Symmetric +/- band on the expected device kernel duration (ms), measured on a Blackhole p150b. Observed
# run-to-run spread is <1%; 5% leaves comfortable headroom for board/thermal variance while still catching a
# real regression. Duration (unlike math util) is not normalized for clock/board, so keep some slack here.
MSA_PERF_MARGIN = 0.05


def _assert_msa_duration(device, run_fn, expected_ms, label):
    """Profile one dispatch of the op with the real-time device profiler and assert its device kernel duration
    is within +/- MSA_PERF_MARGIN of expected_ms. Returns the op output for the shape check."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        # This runs only on the IOMMU-pinned sku (the broad non-IOMMU sdpa glob excludes it by marker), so an
        # inactive profiler means it regressed on a sku that should have it -- fail (not skip), matching the
        # sprint / ring-joint perf checks.
        pytest.fail("Real-time profiler must be active for sparse_sdpa_msa perf checks (needs IOMMU)")

    out, perf_record = profile_realtime_program(device, run_fn)
    duration_ms = perf_record["duration_ns"] / 1e6
    lower = expected_ms * (1 - MSA_PERF_MARGIN)
    upper = expected_ms * (1 + MSA_PERF_MARGIN)
    logger.info(
        f"sparse_sdpa_msa perf {label}: duration={duration_ms:.3f} ms "
        f"(expected {expected_ms:.3f} ms, band [{lower:.3f}, {upper:.3f}])"
    )
    assert lower <= duration_ms <= upper, (
        f"{label} device kernel duration {duration_ms:.3f} ms outside band [{lower:.3f}, {upper:.3f}] ms "
        f"(expected {expected_ms:.3f} ms, margin +/- {MSA_PERF_MARGIN * 100:.0f}%)"
    )
    return out


@run_for_blackhole()
@pytest.mark.parametrize(
    "kv_dtype, expected_ms", [(ttnn.bfloat8_b, 0.774), (ttnn.bfloat16, 1.417)], ids=["bfp8", "bf16"]
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_msa_perf_prod(device, kv_dtype, expected_ms):
    d, H, S, T, topk = 128, 16, 640, 56320, 16
    # Non-causal selection keeps topk=16 valid blocks for every query.
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=7)
    out = _assert_msa_duration(
        device,
        lambda: run_op_msa_native(q, k, v, indices, device, kv_dtype=kv_dtype),
        expected_ms,
        f"prod H={H} n_kv=1 kv_dtype={kv_dtype}",
    )
    assert tuple(out.shape) == (1, H, S, d)


@run_for_blackhole()
@pytest.mark.parametrize(
    "kv_dtype, expected_ms", [(ttnn.bfloat8_b, 3.084), (ttnn.bfloat16, 5.601)], ids=["bfp8", "bf16"]
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_msa_perf_prod_single_chip_gqa(device, kv_dtype, expected_ms):
    d, H, n_kv, S, T, topk = 128, 64, 4, 640, 56320, 16
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=7)
    out = _assert_msa_duration(
        device,
        lambda: run_op_msa_native(q, k, v, indices, device, kv_dtype=kv_dtype),
        expected_ms,
        f"gqa H={H} n_kv={n_kv} kv_dtype={kv_dtype}",
    )
    assert tuple(out.shape) == (1, H, S, d)
