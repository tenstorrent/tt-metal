# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import MathFidelity
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sum_reduce_scalar import (
    FORMATS,
    TILE_DIMENSIONS,
    _num_tiles_for_format,
    test_sum_reduce_scalar,
)


@blackhole_only
@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    num_tiles=_num_tiles_for_format,
    tile_dimensions=TILE_DIMENSIONS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sum_reduce_scalar(
    perf_report,
    formats,
    math_fidelity,
    num_tiles,
    tile_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    test_sum_reduce_scalar(
        formats,
        math_fidelity,
        num_tiles,
        tile_dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
