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

# Device duration in ns per (tokens, topk, n_routed_experts) on a BH P150 (viommu runner), each a median
# of _ITERS dispatches. Recalibrate on the perf runner (DDR-speed dependent): each case logs an "RT-CAL"
# line in this dict's format, so one run regenerates the table. From L2 run 39594798127; the 640-token
# cases repeated within 0.3% across runs 39531468154, 39592189094 and 39594798127.
_EXPECTED_NS: dict[tuple[int, int, int], int] = {
    (640, 4, 128): 15_982,
    (640, 8, 384): 33_595,
    (2560, 8, 256): 47_921,
    (4096, 8, 256): 59_433,
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
        expected_ns=_EXPECTED_NS[(tokens, topk, n_routed_experts)],
        margin=_MARGIN,
        label=f"({tokens}, {topk}, {n_routed_experts})",
        iters=_ITERS,
        verbose=_VERBOSE,
    )
