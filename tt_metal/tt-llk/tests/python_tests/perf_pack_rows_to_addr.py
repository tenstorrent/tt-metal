# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_pack_rows_to_addr import dimension_combinations
from test_pack_rows_to_addr import test_pack_rows_to_addr as _test_pack_rows_to_addr


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Bfp8_b,
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_rows_to_pack=[1, 16, 50, 64],
    dimensions=dimension_combinations,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_pack_rows_to_addr(
    perf_report,
    formats,
    dest_acc,
    num_rows_to_pack,
    dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _test_pack_rows_to_addr(
        formats,
        dest_acc,
        num_rows_to_pack,
        dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
