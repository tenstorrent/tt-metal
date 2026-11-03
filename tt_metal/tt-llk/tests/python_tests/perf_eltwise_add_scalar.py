# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_eltwise_add_scalar import test_eltwise_add_scalar


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    math_fidelity=MathFidelity.LoFi,
    tile_dimensions=[[32, 32], [16, 32], [8, 32]],
    input_dimensions=[[512, 32]],
    output_dimensions=[[128, 32]],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_eltwise_add_scalar(
    perf_report,
    formats,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    test_eltwise_add_scalar(
        formats,
        math_fidelity,
        tile_dimensions,
        input_dimensions,
        output_dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
