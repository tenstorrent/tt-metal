# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import test_sfpu_reduce_sdpa as _func
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation, PerfRunType, ReducePool
from helpers.param_config import (
    input_output_formats,
    parametrize,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ReduceColumn],
    reduce_pool=[ReducePool.Max],  # Only MAX is supported for SDPA reduce
    loop_factor=list(
        range(10, 201, 10)
    ),  # Multiple loop factors to minimize profiler overhead
)
def test_perf_sfpu_reduce_sdpa(
    perf_report,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    loop_factor,
):
    """
    Performance test for SFPU reduce SDPA operation.

    This test specifically measures the performance of the SFPU reduce operation
    used in SDPA (Scaled Dot-Product Attention) implementations. It focuses on
    measuring cycles spent in the SFPU calculations, not including memory operations.

    The test uses a 128x64 input dimension (8 tiles) and performs column-wise
    max reduction, which is the typical operation in SDPA softmax computation.
    """

    input_dimensions = [128, 64]

    # The custom SDPA reduction executes on the pack thread, so PACK_ISOLATE
    # measures the same SFPU sequence as the functional L1_TO_L1 path.
    _func._run_sfpu_reduce_sdpa(
        formats,
        dest_acc,
        mathop,
        reduce_pool,
        input_dimensions,
        loop_factor=loop_factor,
        is_perf=True,
        perf_report=perf_report,
        run_types=[PerfRunType.PACK_ISOLATE],
    )
