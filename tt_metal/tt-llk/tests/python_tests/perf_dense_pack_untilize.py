# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_dense_pack_untilize import test_pack_untilize


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    input_dimensions=[[32, 32], [32, 64], [32, 128], [32, 256], [128, 256]],
    r_dim=[1, 2, 4, 8, 16],
    dest_sync=[DestSync.Half],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_pack_untilize(
    perf_report,
    formats,
    dest_acc,
    input_dimensions,
    r_dim,
    dest_sync,
    run_types,
    loop_factor,
    is_perf,
):
    test_pack_untilize(
        formats,
        dest_acc,
        input_dimensions,
        r_dim,
        dest_sync,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
