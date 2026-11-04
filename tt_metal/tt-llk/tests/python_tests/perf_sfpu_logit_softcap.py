# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES_WITHOUT_MATH_ISOLATE
from test_sfpu_logit_softcap import (
    CAPS,
    FORMATS,
    INPUT_RANGES,
)
from test_sfpu_logit_softcap import test_sfpu_logit_softcap as _test_sfpu_logit_softcap


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    # input_range is stimulus-only; value_bits already records cap.
    input_range=[INPUT_RANGES[0]],
    cap=CAPS,
    **PERF_MEASUREMENT_AXES_WITHOUT_MATH_ISOLATE,
)
def test_perf_sfpu_logit_softcap(
    perf_report, formats, dest_acc, input_range, cap, run_types, loop_factor, is_perf
):
    _test_sfpu_logit_softcap(
        formats,
        dest_acc,
        input_range,
        cap,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


# The polynomial overload is the fused Blaze epilogue. It streams several tiles
# and rounds in the SFPU when dest is BF16, so its rows are not the accurate sweep.
POLYNOMIAL_FORMAT = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
POLYNOMIAL_TILES = 3


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_range=[INPUT_RANGES[0]],
    cap=CAPS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_logit_softcap_polynomial(
    perf_report, dest_acc, input_range, cap, run_types, loop_factor, is_perf
):
    _test_sfpu_logit_softcap(
        POLYNOMIAL_FORMAT,
        dest_acc,
        input_range,
        cap,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        polynomial=True,
        tile_count=POLYNOMIAL_TILES,
    )
