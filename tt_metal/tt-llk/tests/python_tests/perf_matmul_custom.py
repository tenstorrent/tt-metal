# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.device import BootMode
from helpers.format_config import is_dest_acc_needed
from helpers.llk_params import (
    ALL_MATH_FIDELITIES,
    DEST_ACC_MODES,
    DestAccumulation,
    DestSync,
    MathFidelity,
)
from helpers.matmul_sweep import generate_matmul_dimension_combinations
from helpers.param_config import max_dest_tiles, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_matmul_custom import (
    MATMUL_FORMATS,
    ODD_REUSE_DIMS,
    ODD_REUSE_FORMATS,
    THROTTLE_DIMS,
    THROTTLE_FORMATS,
    THROTTLE_LEVELS,
    _run_matmul_custom,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf, case_set):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        perf_case_set=case_set,
    )


@pytest.mark.perf
@parametrize(
    math_fidelity=ALL_MATH_FIDELITIES,
    formats=MATMUL_FORMATS,
    dest_acc=DEST_ACC_MODES,
    dimensions=lambda formats, dest_acc: generate_matmul_dimension_combinations(
        max_dest_tiles(is_dest_acc_needed(formats) or dest_acc == DestAccumulation.Yes)
    ),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom(
    perf_report,
    math_fidelity,
    formats,
    dest_acc,
    dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    input_A_dimensions, input_B_dimensions = dimensions
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=0,
        boot_mode=BootMode.DEFAULT,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf, 0),
    )


@pytest.mark.perf
@parametrize(
    throttle_level=THROTTLE_LEVELS,
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=THROTTLE_FORMATS,
    dest_acc=DEST_ACC_MODES,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_throttle(
    perf_report,
    throttle_level,
    math_fidelity,
    formats,
    dest_acc,
    run_types,
    loop_factor,
    is_perf,
):
    if throttle_level >= 4 and math_fidelity != MathFidelity.LoFi:
        pytest.skip(
            "throttle levels 4/5 do not increment the fidelity phase per LLK; "
            "only correct for LoFi (single-phase)"
        )

    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=BootMode.DEFAULT,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf, 1),
    )


@pytest.mark.perf
@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=ODD_REUSE_FORMATS,
    dimensions=ODD_REUSE_DIMS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_odd_reuse(
    perf_report, math_fidelity, formats, dimensions, run_types, loop_factor, is_perf
):
    input_A_dimensions, input_B_dimensions = dimensions
    _run_matmul_custom(
        math_fidelity,
        formats,
        DestAccumulation.No,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=0,
        boot_mode=BootMode.DEFAULT,
        dest_sync=DestSync.Full,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf, 2),
    )
