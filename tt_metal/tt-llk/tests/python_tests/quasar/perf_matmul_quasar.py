# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DestSync,
    Transpose,
)
from helpers.matmul_sweep import dest_corner_mn
from helpers.param_config import (
    max_dest_tiles,
    parametrize,
    quasar_dest_sync_modes,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from quasar.test_matmul_quasar import (
    FLOAT32_MATMUL_FORMATS,
    FULL_MATMUL_SHAPES,
    MATMUL_FORMAT,
    NON_MX_MATMUL_FORMATS,
    PERF_KT_DIMS,
    TINY_MATMUL_SHAPE_CASES,
    float32_matmul_dest_acc_modes,
    matmul_dest_acc_modes,
    matmul_enable_direct_indexing,
    matmul_exact_fill_ct_dims,
    matmul_perf_rt_dims,
    matmul_register_format_hints,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul

# The 32-bit pairs test_matmul_float32 checks functionally; perf measures exactly those.
PERF_MATMUL_FORMAT = MATMUL_FORMAT + FLOAT32_MATMUL_FORMATS

# WH/BH perf_matmul also measures a long K (kt=32), but only where the A operand does not
# stride per output row, i.e. a single output row (rt=1).
PERF_LONG_KT = 32


def _is_mx(formats) -> bool:
    return formats.input_format.is_mx_format() or formats.output_format.is_mx_format()


def _fits_perf_l1(formats, rt_dim: int, ct_dim: int, kt_dim: int) -> bool:
    """A, B and the result fit between the perf stimuli address and the perf counters."""
    stimuli_bytes = (
        rt_dim * kt_dim * formats.input_format.num_bytes_per_tile()
        + kt_dim * ct_dim * formats.input_format_B.num_bytes_per_tile()
        + rt_dim * ct_dim * formats.output_format.num_bytes_per_tile()
    )
    return (
        StimuliConfig.STIMULI_L1_ADDRESS_PERF + stimuli_bytes
        <= TestConfig.PERF_COUNTERS_BASE_ADDR
    )


def perf_matmul_dest_acc_modes(formats):
    if formats in FLOAT32_MATMUL_FORMATS:
        return float32_matmul_dest_acc_modes(formats)
    return matmul_dest_acc_modes(formats)


def perf_matmul_dest_sync_modes(formats):
    """Non-MX formats also sweep Full to pair with WH/BH perf_matmul; MX stays Half."""
    if _is_mx(formats):
        return quasar_dest_sync_modes(is_perf=True)
    return [DestSync.Half, DestSync.Full]


def _dest_corners(dest_acc, dest_sync):
    return dest_corner_mn(max_dest_tiles(dest_acc, dest_sync))


def perf_matmul_rt_dims(formats, dest_acc, dest_sync):
    """Non-MX formats use the WH/BH perf_matmul dest corners; MX keeps exact fill."""
    if _is_mx(formats):
        return matmul_perf_rt_dims(dest_acc, dest_sync)
    return list(dict.fromkeys(rt for rt, _ in _dest_corners(dest_acc, dest_sync)))


def perf_matmul_ct_dims(formats, dest_acc, dest_sync, rt_dim):
    if _is_mx(formats):
        return matmul_exact_fill_ct_dims(dest_acc, dest_sync, rt_dim)
    return [ct for rt, ct in _dest_corners(dest_acc, dest_sync) if rt == rt_dim]


def perf_matmul_kt_dims(formats, rt_dim, ct_dim):
    """PERF_KT_DIMS, plus PERF_LONG_KT for non-MX single-row blocks that fit in L1."""
    if _is_mx(formats):
        return PERF_KT_DIMS
    kt_dims = PERF_KT_DIMS + ([PERF_LONG_KT] if rt_dim == 1 else [])
    return [kt for kt in kt_dims if _fits_perf_l1(formats, rt_dim, ct_dim, kt)]


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(FULL_MATMUL_SHAPES),
    formats=PERF_MATMUL_FORMAT,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    dest_sync=perf_matmul_dest_sync_modes,
    dest_acc=perf_matmul_dest_acc_modes,
    rt_dim=runtime(perf_matmul_rt_dims),
    ct_dim=runtime(perf_matmul_ct_dims),
    kt_dim=runtime(perf_matmul_kt_dims),
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
