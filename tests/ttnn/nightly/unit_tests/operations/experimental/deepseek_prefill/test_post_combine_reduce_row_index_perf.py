# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time gate for post_combine_reduce's per-token weights and row-indexed combine, measured with the real-time
program profiler on one chip at gpt-oss-20b's routed-prefill shapes (2048 tokens, top-4 of 32 experts, 8 local,
emb 2880).

The two input forms save work outside the op, so the test times the whole sequence a caller runs before and after:

- gathered (default forms): ttnn.embedding gathers each token's k expert rows into a [T, k, emb] copy in L1 (the
  other chips' slots, which all point at the spare row, served from a per-core L1 copy of it via padding_idx), the
  [T, k] router weights are reshaped to [T, k, 1] (one weight per page), and post_combine_reduce reduces the copy;
- indexed: post_combine_reduce reads each slot's row of the expert output table through combine_row_index and the
  [T, k] weights as they are.

The gate is the ratio of the two device times, so it does not depend on the board's absolute speed; each case also
logs both times ("RT-CAL").

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses 64-bit PCIe
addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

T = 2048  # tokens
K = 4  # experts per token
E = 32  # routed experts
LOCAL = 8  # experts on this chip
EMB = 2880
_ITERS = 3
# Indexed sequence / gathered sequence device time must not exceed this. Measured 2026-10-26 on one Blackhole chip of
# a QuietBox 2 (two runs): gathered 223.6 / 225.2 us, indexed 91.1 / 91.5 us, ratio 0.407 / 0.407.
_MAX_RATIO = 0.5


def _device_ns(device, run):
    """Total device time of the programs one call of run() dispatches (the sum over _ITERS calls / _ITERS)."""

    def run_all():
        for _ in range(_ITERS):
            run()

    _, per_program = profile_realtime_program_merged(device, run_all)
    return sum(entry["duration_ns"] for entry in per_program.values()) / _ITERS


@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="perf gate measured on Blackhole")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_row_indexed_combine_perf(device):
    require_realtime_profiler("post_combine_reduce row-index perf checks")
    torch.manual_seed(13)
    indices = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    local = indices < LOCAL
    pairs = int(local.sum())
    spare = pairs
    rows = -(-(pairs + 1) // 32) * 32
    row_index = torch.full((T, K), spare, dtype=torch.int64)
    row_index[local] = torch.randperm(pairs)  # the expert output's rows are in expert order, not token order

    def dev(t, dtype=None):
        return ttnn.from_torch(
            t, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    table = dev(torch.randn(rows, EMB).to(torch.bfloat16))
    weights = dev(torch.softmax(torch.randn(T, K), dim=-1).to(torch.bfloat16))  # [T, K], as the router writes them
    pos = dev(row_index.to(torch.int32), ttnn.uint32)  # [T, K]
    common = dict(
        indices=dev(indices.reshape(1, 1, T, K).to(torch.uint16), ttnn.uint16),
        expert_dispatch_table=dev(torch.where(torch.arange(E) < LOCAL, 0, -1).to(torch.int32)),
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def gathered():
        combined = ttnn.embedding(
            pos, table, padding_idx=spare, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        combined = ttnn.reshape(combined, (1, 1, T, K, EMB))
        w = ttnn.reshape(weights, (1, 1, T, K, 1))
        out = ttnn.experimental.deepseek_prefill.post_combine_reduce(combined, w, expert_dim=3, **common)
        ttnn.deallocate(combined)
        return out

    def indexed():
        return ttnn.experimental.deepseek_prefill.post_combine_reduce(
            table,
            ttnn.reshape(weights, (1, 1, T, K)),
            expert_dim=3,
            combine_row_index=ttnn.reshape(pos, (1, 1, T, K)),
            skip_row=spare,
            **common,
        )

    # warm up (compile) both; the two forms are bit-identical
    a = ttnn.to_torch(gathered()).reshape(T, EMB)
    b = ttnn.to_torch(indexed()).reshape(T, EMB)
    assert torch.equal(a, b), "row-indexed combine differs from the gathered combine"

    gathered_ns = _device_ns(device, gathered)
    indexed_ns = _device_ns(device, indexed)
    ratio = indexed_ns / gathered_ns
    logger.info(f"RT-CAL gathered {gathered_ns / 1e3:.1f} us, indexed {indexed_ns / 1e3:.1f} us, ratio {ratio:.3f}")
    assert (
        ratio <= _MAX_RATIO
    ), f"indexed sequence {indexed_ns:.0f} ns is {ratio:.3f} of the gathered {gathered_ns:.0f} ns"
