# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import is_4row_arch
from helpers.constraints import get_valid_math_fidelities
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    MathFidelity,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import parametrize, quasar_mx_smoke
from helpers.reduce_perf_matrix import (
    BLOCK_LENGTH_REPRESENTATIVE,
    BLOCK_LENGTH_TILE_COUNTS,
    CORE_FORMATS,
    CORE_REDUCE_DIMS,
    CORE_TILE_DIMENSIONS,
    PERF_POOL_TYPES,
    PERF_TILE_COUNT,
    SCALAR_REPRESENTATIVE,
    SHARED_EXTENSION_FORMATS,
    perf_input_dimensions,
    perf_math_fidelities,
)
from quasar.test_reduce_quasar import (
    reduce_dest_acc_modes,
    reduce_dest_sync_modes,
    reduce_is_reduce_to_one_modes,
    reduce_tile_dimensions,
)
from quasar.test_reduce_quasar import test_reduce_quasar as run_reduce_quasar
from quasar.test_reduce_quasar import (
    test_reduce_quasar_mxfp4_2x_gapool as run_reduce_quasar_mxfp4_2x_gapool,
)

# Quasar has no Bfp8_b, so its extension is the shared pairs plus the MX smoke pair.
REDUCE_PERF_FORMATS = (
    CORE_FORMATS
    + SHARED_EXTENSION_FORMATS
    + quasar_mx_smoke(DataFormat.MxFp4, DataFormat.Float16_b)
)

PERF_MEASUREMENT = dict(
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=True),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)


def _run_reduce_quasar_perf(
    perf_report,
    formats,
    tile_dimensions,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    dest_sync_mode,
    run_types,
    loop_factor,
    is_perf,
    tile_count=PERF_TILE_COUNT,
):
    run_reduce_quasar(
        formats,
        tile_dimensions,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        dest_sync_mode,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        input_dimensions=perf_input_dimensions(tile_dimensions, tile_count),
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.nightly
@parametrize(
    formats=REDUCE_PERF_FORMATS,
    tile_dimensions=lambda formats: reduce_tile_dimensions(formats, is_perf=True),
    dest_acc=reduce_dest_acc_modes,
    reduce_dim=CORE_REDUCE_DIMS,
    pool_type=PERF_POOL_TYPES,
    # Core formats keep both fidelities so they pair with WH/BH; init lowers Float16_b itself.
    math_fidelity=lambda formats, pool_type: perf_math_fidelities(
        pool_type,
        None if formats in CORE_FORMATS else get_valid_math_fidelities(formats),
    ),
    is_reduce_to_one=reduce_is_reduce_to_one_modes,
    **PERF_MEASUREMENT,
)
def test_perf_reduce_quasar(
    perf_report,
    formats,
    tile_dimensions,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    dest_sync_mode,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_quasar_perf(
        perf_report,
        formats,
        tile_dimensions,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        dest_sync_mode,
        run_types,
        loop_factor,
        is_perf,
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.nightly
@parametrize(
    **SCALAR_REPRESENTATIVE,
    tile_dimensions=CORE_TILE_DIMENSIONS,
    is_reduce_to_one=[False],
    **PERF_MEASUREMENT,
)
def test_perf_reduce_scalar_quasar(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    tile_dimensions,
    is_reduce_to_one,
    dest_sync_mode,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_quasar_perf(
        perf_report,
        formats,
        tile_dimensions,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        dest_sync_mode,
        run_types,
        loop_factor,
        is_perf,
    )


# HiFi4 past the Float16_b LoFi cap, as for the core rows, so these pair with WH/BH.
@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.nightly
@parametrize(
    **BLOCK_LENGTH_REPRESENTATIVE,
    math_fidelity=[MathFidelity.HiFi4],
    tile_dimensions=CORE_TILE_DIMENSIONS,
    is_reduce_to_one=[False],
    tile_count=BLOCK_LENGTH_TILE_COUNTS,
    **PERF_MEASUREMENT,
)
def test_perf_reduce_block_length_quasar(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    tile_dimensions,
    is_reduce_to_one,
    tile_count,
    dest_sync_mode,
    run_types,
    loop_factor,
    is_perf,
):
    _run_reduce_quasar_perf(
        perf_report,
        formats,
        tile_dimensions,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        dest_sync_mode,
        run_types,
        loop_factor,
        is_perf,
        tile_count=tile_count,
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.skipif(
    is_4row_arch(),
    reason="MxFp4_2x GAPOOL reduce is not implemented on the 4-row Quasar architecture",
)
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
    dest_acc=reduce_dest_acc_modes,
    reduce_dim=[ReduceDimension.Column],
    pool_type=[ReducePool.Sum, ReducePool.Average],
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=True),
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
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
