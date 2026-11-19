# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PERF_RUN_TYPES_QUASAR, Transpose
from helpers.param_config import parametrize, quasar_dest_sync_modes, runtime
from quasar.test_matmul_quasar import (
    FULL_MATMUL_SHAPES,
    MATMUL_FORMAT,
    NON_MX_MATMUL_FORMATS,
    PERF_KT_DIMS,
    TINY_MATMUL_SHAPE_CASES,
    matmul_dest_acc_modes,
    matmul_enable_direct_indexing,
    matmul_exact_fill_ct_dims,
    matmul_perf_rt_dims,
    matmul_register_format_hints,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(FULL_MATMUL_SHAPES),
    formats=MATMUL_FORMAT,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    dest_sync=lambda: quasar_dest_sync_modes(is_perf=True),
    dest_acc=matmul_dest_acc_modes,
    rt_dim=runtime(matmul_perf_rt_dims),
    ct_dim=runtime(matmul_exact_fill_ct_dims),
    kt_dim=runtime(PERF_KT_DIMS),
    register_format_hint=matmul_register_format_hints,
    enable_direct_indexing=matmul_enable_direct_indexing,
    unpack_transpose_faces=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_quasar(
    perf_report,
    input_tile_dimensions,
    ct_dim,
    rt_dim,
    kt_dim,
    math_fidelity,
    dest_sync,
    dest_acc,
    formats,
    register_format_hint,
    enable_direct_indexing,
    unpack_transpose_faces,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        input_tile_dimensions,
        ct_dim,
        rt_dim,
        kt_dim,
        math_fidelity,
        dest_sync,
        dest_acc,
        formats,
        register_format_hint,
        enable_direct_indexing,
        unpack_transpose_faces,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(TINY_MATMUL_SHAPE_CASES),
    formats=NON_MX_MATMUL_FORMATS,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    dest_sync=lambda: quasar_dest_sync_modes(is_perf=True),
    dest_acc=matmul_dest_acc_modes,
    rt_dim=runtime(matmul_perf_rt_dims),
    ct_dim=runtime(matmul_exact_fill_ct_dims),
    kt_dim=runtime(PERF_KT_DIMS),
    register_format_hint=matmul_register_format_hints,
    enable_direct_indexing=matmul_enable_direct_indexing,
    unpack_transpose_faces=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_tiny_quasar(
    perf_report,
    input_tile_dimensions,
    math_fidelity,
    dest_sync,
    dest_acc,
    ct_dim,
    rt_dim,
    kt_dim,
    formats,
    register_format_hint,
    enable_direct_indexing,
    unpack_transpose_faces,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        input_tile_dimensions,
        ct_dim,
        rt_dim,
        kt_dim,
        math_fidelity,
        dest_sync,
        dest_acc,
        formats,
        register_format_hint,
        enable_direct_indexing,
        unpack_transpose_faces,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
