# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for ttnn.scatter on long ROW_MAJOR rows, measured with the real-time program profiler.

Each row's first phase puts the input chunk into the output page before the scatter writes into it. That copy is
what this gates: on Wormhole / Blackhole the NoC reads the chunk straight into the output page; the previous
element-by-element copy out of a scratchpad took ~300 us for one 33K-element fp32 row on Blackhole, more than the
scatter itself. The largest case is the shape gpt-oss-20b's routed prefill scatters per layer at 8K tokens (one fp32
row of 4 x 8192 pair slots plus padding, 32768 indices).

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses 64-bit PCIe
addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import (
    assert_op_duration_merged,
    profile_realtime_program,
    require_realtime_profiler,
)

# RT records carry kernel sources, not an OP CODE, so identify the op by its reader kernel. The
# float32 rows run the native kernel; a bfloat16 row with a uint32 index routes to the codegen
# ROW_MAJOR kernel on Blackhole (ttnn::scatter's routing), so its row names that reader.
_OP_KERNELS = {
    "native": "/scatter/device/kernels/dataflow/reader_scatter.cpp",
    "codegen": "/scatter/codegen/kernels/scatter_reader_rm.cpp",
}

# Median of this many dispatches (a single dispatch throws occasional flyers).
_ITERS = 5

# Log every program in the profiled window -- use when recalibrating.
_VERBOSE = False

_MARGIN = 0.10

# Device duration in ns per (dtype, input row length, index row length): one Blackhole chip of a QuietBox 2,
# 2026-10-26, median of _ITERS dispatches (two runs agreed within 0.01%). The previous element-by-element copy measured
# 199_690 / 1_576_617 / 1_723_962 ns, outside the band. Recalibrate on the perf runner: each case logs an "RT-CAL" line
# with its measured median.
_EXPECTED_NS: dict[tuple[str, int, int, str], int] = {
    ("float32", 4224, 4096, "native"): 165_207,
    ("float32", 33024, 32768, "native"): 1_307_336,
    # The bfloat16 row on the codegen ROW_MAJOR kernel: one Blackhole P150 chip, 2026-11-19, median of _ITERS
    # (the native kernel measured 1_454_746 ns on this row before the routing change).
    ("bfloat16", 33024, 32768, "codegen"): 598_746,
}

_CASES = [
    ("float32", 4224, 4096, "native"),
    ("float32", 33024, 32768, "native"),
    ("bfloat16", 33024, 32768, "codegen"),
]

_DTYPES = {"float32": (ttnn.float32, torch.float32), "bfloat16": (ttnn.bfloat16, torch.bfloat16)}


@pytest.mark.parametrize(
    "dtype_name, input_len, index_len, route",
    [pytest.param(*case, id=f"{case[0]}-{case[1]}x{case[2]}-{case[3]}-perf") for case in _CASES],
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="perf baselines are Blackhole-specific")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_scatter_perf(device, dtype_name, input_len, index_len, route):
    require_realtime_profiler("scatter perf checks")
    ttnn_dtype, torch_dtype = _DTYPES[dtype_name]

    torch.manual_seed(0)
    torch_input = torch.randn(1, input_len).to(torch_dtype)
    torch_index = torch.randperm(input_len)[:index_len].reshape(1, index_len)
    torch_source = torch.randn(1, index_len).to(torch_dtype)

    def to_device(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_input = to_device(torch_input, ttnn_dtype)
    tt_index = to_device(torch_index, ttnn.uint32)
    tt_source = to_device(torch_source, ttnn_dtype)

    # correctness first, so a case that gets fast by computing the wrong thing fails here
    # (exact: the op copies and overwrites values, no arithmetic). It runs in its own profiler window: the profiler
    # delivers records asynchronously, so without one this dispatch's record could land in the timed window below.
    expected = torch.scatter(torch_input, -1, torch_index, torch_source)
    run = lambda: ttnn.scatter(tt_input, -1, tt_index, tt_source)
    actual, _ = profile_realtime_program(device, lambda: ttnn.to_torch(run()), collect_all=True)
    assert torch.equal(
        actual, expected
    ), f"scatter output differs from the reference for {dtype_name}, {input_len}, {index_len}"

    label = f'("{dtype_name}", {input_len}, {index_len}, "{route}")'
    expected_ns = _EXPECTED_NS.get((dtype_name, input_len, index_len, route))
    if expected_ns is None:
        pytest.skip(f"no baseline for {label}; add it to _EXPECTED_NS from the RT-CAL line")
    assert_op_duration_merged(
        device,
        run,
        _OP_KERNELS[route],
        expected_ns=expected_ns,
        margin=_MARGIN,
        label=label,
        iters=_ITERS,
        verbose=_VERBOSE,
    )
