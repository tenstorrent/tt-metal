# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MATH_ISOLATE perf for the EMA SFPU kernel (ckernel_sfpu_ema.h).

The TILE_LOOP marker under MATH_ISOLATE covers the math pipe with no dest
handshake with pack.

It does still include the datacopy that feeds Dest in the 16-bit forms: it retires the SrcA valid bits unpack sets,
so mean(MATH_ISOLATE) is the SFPU block plus a fixed datacopy that cancels in a delta, not the kernel alone.

cycles/tile lands in the TILE_LOOP row of the .post.csv as mean(MATH_ISOLATE).

Both DEST widths are swept; the Float32 form unpacks straight to Dest and has no datacopy.
"""


import pytest
from helpers.param_config import parametrize
from helpers.perf.core import PerfRunType
from test_sfpu_ema import (
    EMA_ALPHA,
    EMA_BETA,
    SFPU_EMA_PERF_SWEEP,
    _run_sfpu_ema,
)


@pytest.mark.perf
@parametrize(
    **SFPU_EMA_PERF_SWEEP,
    run_types=[
        [
            PerfRunType.MATH_ISOLATE,
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ]
    ],
    loop_factor=[16],  # amortise profiler overhead
    input_dimensions=[[128, 64]],  # tile_cnt: 8
    is_perf=[True],
)
def test_perf_sfpu_ema(
    perf_report,
    formats,
    dest_acc,
    run_types,
    loop_factor,
    input_dimensions,
    is_perf,
):
    _run_sfpu_ema(
        EMA_ALPHA,
        EMA_BETA,
        dest_acc,
        formats=formats,
        input_dimensions=input_dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
