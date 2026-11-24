# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_matmul_in1_column_stride import (
    test_matmul_in1_column_stride_and_restore as _test_matmul_in1_column_stride_and_restore,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    kt_dim=[3, 20],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_matmul_in1_column_stride_and_restore(
    perf_report, formats, dest_acc, kt_dim, run_types, loop_factor, is_perf
):
    _test_matmul_in1_column_stride_and_restore(
        formats,
        dest_acc,
        kt_dim,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
