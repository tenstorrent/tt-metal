# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_fast_tilize_metal_api import (
    test_fast_tilize_metal_api as _test_fast_tilize_metal_api,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    dimensions=[(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (10, 12)],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_fast_tilize_metal_api(
    perf_report, formats, dest_acc, dimensions, run_types, loop_factor, is_perf
):
    _test_fast_tilize_metal_api(
        formats,
        dest_acc,
        dimensions,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
