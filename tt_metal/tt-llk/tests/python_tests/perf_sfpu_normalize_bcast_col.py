# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# SFPU column normalize (full tile, two tiles, residual), in place, per format.

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_sfpu_normalize_bcast_col import run_norm_col


@pytest.mark.perf
@parametrize(
    case=[
        f"{v}-{f}-A_in-rand-rt0" for v in ("FT", "TT", "RES") for f in ("F32", "BF16")
    ],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
)
def test_perf_sfpu_normalize_bcast_col(perf_report, case, run_types, loop_factor):
    run_norm_col(
        case,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
