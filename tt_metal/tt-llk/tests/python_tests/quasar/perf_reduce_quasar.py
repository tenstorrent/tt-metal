# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    PERF_RUN_TYPES_QUASAR,
    MathFidelity,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import parametrize
from quasar.test_reduce_quasar import (
    REDUCE_FORMATS,
    reduce_dest_acc_modes,
    reduce_dest_sync_modes,
    reduce_implied_math_formats,
    reduce_pool_type_and_math_fidelity_combinations,
)
from quasar.test_reduce_quasar import test_reduce_quasar as run_reduce_quasar
from quasar.test_reduce_quasar import (
    test_reduce_quasar_mxfp4_2x_gapool as run_reduce_quasar_mxfp4_2x_gapool,
)


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.nightly
@parametrize(
    formats=REDUCE_FORMATS,
    tile_dimensions=[(32, 32)],
    dest_acc=lambda: reduce_dest_acc_modes(is_perf=True),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type_and_math_fidelity=lambda: reduce_pool_type_and_math_fidelity_combinations(
        is_perf=True
    ),
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=True),
    implied_math_format=lambda formats: reduce_implied_math_formats(
        formats, is_perf=True
    ),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_reduce_quasar(
    perf_report,
    formats,
    tile_dimensions,
    dest_acc,
    reduce_dim,
    pool_type_and_math_fidelity,
    dest_sync_mode,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    run_reduce_quasar(
        formats,
        tile_dimensions,
        dest_acc,
        reduce_dim,
        pool_type_and_math_fidelity,
        dest_sync_mode,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    register_format_hint=[DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B],
    formats=lambda register_format_hint: [
        InputOutputFormat(
            DataFormat.MxFp4,
            DataFormat.Float16,
            register_format_hint=register_format_hint,
        ),
        InputOutputFormat(
            DataFormat.MxFp4,
            DataFormat.Float16_b,
            register_format_hint=register_format_hint,
        ),
    ],
    dest_acc=lambda: reduce_dest_acc_modes(is_perf=True),
    reduce_dim=[ReduceDimension.Column],
    pool_type=[ReducePool.Sum, ReducePool.Average],
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=True),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_reduce_quasar_mxfp4_2x_gapool(
    perf_report,
    register_format_hint,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    dest_sync_mode,
    run_types,
    loop_factor,
    is_perf,
):
    run_reduce_quasar_mxfp4_2x_gapool(
        register_format_hint,
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        dest_sync_mode,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
