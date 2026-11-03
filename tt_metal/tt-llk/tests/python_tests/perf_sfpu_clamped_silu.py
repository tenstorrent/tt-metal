# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_clamped_silu import (
    ALL_OPS,
    FORMATS,
    _op_ranges,
    _op_scalars,
    test_sfpu_clamped_silu,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    op=ALL_OPS,
    scalars=lambda op: _op_scalars(op),
    # One stimulus range per op. The ranges are not a recorded kernel column.
    input_range=lambda op: [_op_ranges(op)[0]],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_sfpu_clamped_silu(
    perf_report,
    formats,
    dest_acc,
    op,
    scalars,
    input_range,
    run_types,
    loop_factor,
    is_perf,
):
    test_sfpu_clamped_silu(
        formats,
        dest_acc,
        op,
        scalars,
        input_range,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
