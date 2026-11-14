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
    matmul_register_format_hints,
    matmul_tile_dimensions,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul

# The 32-bit pairs test_matmul_float32 checks functionally; perf measures exactly those.
PERF_MATMUL_FORMAT = MATMUL_FORMAT + FLOAT32_MATMUL_FORMATS

# WH/BH perf_matmul also measures a long K (kt=32), but only where the A operand does not
# stride per output row, i.e. a single output row (rt=1).
PERF_LONG_KT = 32

# Stimuli start at the perf input address and must end below the perf-counter region
# (tests/helpers/include/perf.h); A, B and the result share that window.
_PERF_STIMULI_BASE = 0x21000
_PERF_STIMULI_END = 0x169000


def _is_32_bit_pair(format) -> bool:
    return format in FLOAT32_MATMUL_FORMATS


def _is_mx(format) -> bool:
    return format.input_format.is_mx_format() or format.output_format.is_mx_format()


def _fits_perf_l1(format, rt_dim: int, ct_dim: int, kt_dim: int) -> bool:
    """A (rt*kt) and B (kt*ct) in the input format and the result (rt*ct) fit the stimuli window."""
    stimuli_bytes = (
        rt_dim * kt_dim * format.input_format.num_bytes_per_tile()
        + kt_dim * ct_dim * format.input_format_B.num_bytes_per_tile()
        + rt_dim * ct_dim * format.output_format.num_bytes_per_tile()
    )
    return _PERF_STIMULI_BASE + stimuli_bytes <= _PERF_STIMULI_END


def perf_matmul_dest_acc_modes(format):
    """32-bit pairs follow test_matmul_float32: a Float32 output needs a 32-bit Dest."""
    if _is_32_bit_pair(format):
        return float32_matmul_dest_acc_modes(format)
    return matmul_dest_acc_modes(format)


def perf_matmul_dest_sync_modes(format):
    """Non-MX formats sweep Half and Full to pair with WH/BH perf_matmul; MX stays Half."""
    if _is_mx(format):
        return quasar_dest_sync_modes(is_perf=True)
    return [DestSync.Half, DestSync.Full]


def perf_matmul_tile_dims(format, dest_acc, dest_sync_mode):
    """(ct, rt, kt). Non-MX formats use the WH/BH perf_matmul dest corners; MX keeps exact fill.

    kt is PERF_KT_DIMS ([1, 4]) everywhere, plus PERF_LONG_KT for single-row (rt=1) blocks,
    the long-K rows WH/BH keep under their 16-tile perf ring. Quasar reads operands from
    real L1 buffers, so a long K is bounded by L1 instead (``_fits_perf_l1``).
    """
    if _is_mx(format):
        return matmul_tile_dimensions(
            dest_acc, dest_sync_mode, exact_dest_fill=True, is_perf=True
        )
    return [
        (ct_dim, rt_dim, kt_dim)
        for rt_dim, ct_dim in dest_corner_mn(max_dest_tiles(dest_acc, dest_sync_mode))
        for kt_dim in (*PERF_KT_DIMS, PERF_LONG_KT)
        if (kt_dim != PERF_LONG_KT or rt_dim == 1)
        and _fits_perf_l1(format, rt_dim, ct_dim, kt_dim)
    ]


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(FULL_MATMUL_SHAPES),
    format=PERF_MATMUL_FORMAT,
    math_fidelity=lambda format: get_valid_math_fidelities(format),
    dest_sync_mode=perf_matmul_dest_sync_modes,
    dest_acc=perf_matmul_dest_acc_modes,
    matmul_tile_dims=runtime(perf_matmul_tile_dims),
    register_format_hint=matmul_register_format_hints,
    enable_direct_indexing=matmul_enable_direct_indexing,
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_quasar(
    perf_report,
    input_tile_dimensions,
    matmul_tile_dims,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        input_tile_dimensions,
        matmul_tile_dims,
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        format,
        register_format_hint,
        enable_direct_indexing,
        transpose,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(TINY_MATMUL_SHAPE_CASES),
    format=NON_MX_MATMUL_FORMATS,
    math_fidelity=lambda format: get_valid_math_fidelities(format),
    dest_sync_mode=lambda: quasar_dest_sync_modes(is_perf=True),
    dest_acc=matmul_dest_acc_modes,
    matmul_tile_dims=runtime(
        lambda dest_acc, dest_sync_mode: matmul_tile_dimensions(
            dest_acc,
            dest_sync_mode,
            exact_dest_fill=True,
            is_perf=True,
        )
    ),
    register_format_hint=matmul_register_format_hints,
    enable_direct_indexing=matmul_enable_direct_indexing,
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_tiny_quasar(
    perf_report,
    input_tile_dimensions,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    matmul_tile_dims,
    format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        input_tile_dimensions,
        matmul_tile_dims,
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        format,
        register_format_hint,
        enable_direct_indexing,
        transpose,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
