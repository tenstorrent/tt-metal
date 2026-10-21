# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for ttnn.experimental.deepseek_prefill.masked_bincount over the per-chip token
counts MoE prefill runs at, measured with the real-time program profiler.

The op sizes its core grid from the token count (the fewest cores that keep each at <= 128 rows), so
these cases pin the grids prefill actually takes: 640 tokens -> 5 cores (Mistral-Small-4, Kimi-K2.7),
2560 -> 20, 4096 -> 32.

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses
64-bit PCIe addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler

# RT records carry kernel sources, not an OP CODE, so identify the op by its kernel directory.
_OP_KERNEL_DIR = "/masked_bincount/"

# Median of this many dispatches: a single one throws occasional flyers, and the op is short enough
# (tens of us) that one flyer would move a per-dispatch number past the margin.
_ITERS = 5

# Log every program in the profiled window, per iteration -- use when recalibrating.
_VERBOSE = False

# The op is short, so fixed launch and tree-reduction overhead is a large share of it.
_MARGIN = 0.10
# margin=1.0 zeroes the floor and leaves only a ceiling at 2x: for a case without a P150 baseline yet,
# so the gate still catches a large regression without failing on an uncalibrated number.
_CEILING_ONLY = 1.0

# (device duration in ns, margin) per (tokens, topk, n_routed_experts) on a BH P150 (viommu runner).
# Recalibrate on the perf runner (DDR-speed dependent): each case logs an "RT-CAL" line in this dict's
# format, so one run regenerates the table. The 640-token cases are medians of 5 from L2 runs
# 39531468154 / 39592189094 (they agree within 0.3%); the others are estimates, ceiling-only until a
# P150 run measures them.
_EXPECTED_NS: dict[tuple[int, int, int], tuple[int, float]] = {
    (640, 4, 128): (16_033, _MARGIN),
    (640, 8, 384): (33_587, _MARGIN),
    (2560, 8, 256): (60_000, _CEILING_ONLY),
    (4096, 8, 256): (80_000, _CEILING_ONLY),
}


def _golden(indices, mask, n_routed_experts, num_experts_per_token):
    selected = indices[:, :num_experts_per_token].reshape(-1)
    counts = torch.bincount(selected, minlength=n_routed_experts + 1)[:n_routed_experts]
    return counts * (mask[:n_routed_experts] >= 0).to(counts.dtype)


@pytest.mark.parametrize(
    "tokens, topk, n_routed_experts",
    list(_EXPECTED_NS),
    ids=["isl640-mistral-5cores", "isl640-kimi-5cores", "isl2560-20cores", "isl4096-32cores"],
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="baselines are Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_masked_bincount_perf(device, tokens: int, topk: int, n_routed_experts: int):
    require_realtime_profiler("masked_bincount perf checks")
    expected_ns, margin = _EXPECTED_NS[(tokens, topk, n_routed_experts)]

    torch.manual_seed(0)
    indices = torch.randint(0, n_routed_experts, (tokens, topk), dtype=torch.int64)
    # Half the experts present, as on a chip that holds half of its dispatch group's experts.
    mask = torch.where(torch.arange(n_routed_experts) % 2 == 0, 0, -1).to(torch.int32)
    tt_indices = ttnn.from_torch(indices.to(torch.int16), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def run():
        return ttnn.experimental.deepseek_prefill.masked_bincount(tt_indices, tt_mask, n_routed_experts, topk)

    # Check correctness first, so a case that gets fast by counting the wrong thing fails here rather
    # than passing the band.
    actual = ttnn.to_torch(run()).reshape(-1).to(torch.int64)
    assert torch.equal(actual, _golden(indices, mask, n_routed_experts, topk))

    assert_op_duration_merged(
        device,
        run,
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=f"({tokens}, {topk}, {n_routed_experts})",
        iters=_ITERS,
        verbose=_VERBOSE,
    )
