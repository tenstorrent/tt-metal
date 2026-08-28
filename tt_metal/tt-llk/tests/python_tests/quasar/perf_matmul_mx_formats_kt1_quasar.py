# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul performance across every MX L1 data format on Quasar, with kt=1.

Same sweep as perf_matmul_mx_formats_quasar.py, but with a single K tile: the
output block stays a dest-full 1x8 tiles while the operands shrink from
[32,128] x [128,256] (36 input tiles) to [32,32] x [32,256] (9 input tiles).

A normalisation step still covers 32x32x32 = 32768 MAC, so per-step figures stay
comparable to the kt=4 report. What changes is that ``tile_cnt`` drops from 32 to
8, so each step now carries one whole packed output tile instead of a quarter of
one, and packing cost per step rises accordingly.

Kept in its own module so its report lands in
perf_data/perf_matmul_mx_formats_kt1_quasar/ rather than overwriting the kt=4
results.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from quasar.test_matmul_quasar import test_matmul as run_matmul

# Every MX format that exists as an L1 format, with its tile footprint in bytes
# (1024 elements plus one shared exponent per 32-element block):
#   MxFp8R  E5M2 + block exp   1056 B
#   MxFp8P  E4M3 + block exp   1056 B
#   MxInt8  S1.6 + block exp   1056 B
#   MxFp4   E2M1 + block exp    544 B
#   MxInt4  S1.2 + block exp    544 B
#   MxInt2  S1.0 + block exp    288 B
MX_FORMATS = [
    DataFormat.MxFp8R,
    DataFormat.MxFp8P,
    DataFormat.MxInt8,
    DataFormat.MxFp4,
    DataFormat.MxInt4,
    DataFormat.MxInt2,
]

# Dest-full 1x8 output tiles, accumulated over a single K tile.
MT_DIM, NT_DIM, KT_DIM = 1, 8, 1
DIMENSIONS = [
    [MT_DIM * TILE_DIM, KT_DIM * TILE_DIM],
    [KT_DIM * TILE_DIM, NT_DIM * TILE_DIM],
]


def mx_register_format_hint(format):
    """Pin the src-register format the unpacker must produce for this L1 format.

    MxFp4 is measured on the 2x path: the unpacker packs two sub-elements per src
    register lane, which lets the matmul MOP cover a tile in 8 MVMULs instead of
    16. Pin the MxFp4_2x_B (Float16_b exponent family) variant so every format in
    the sweep ends up in the same exponent family in the src registers. Every
    other MX format unpacks straight to Float16_b and needs no hint.
    """
    return (
        [DataFormat.MxFp4_2x_B] if format.input_format == DataFormat.MxFp4 else [None]
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=input_output_formats(MX_FORMATS, same=True),
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    dimensions=[DIMENSIONS],
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=mx_register_format_hint,
    enable_direct_indexing=[False],
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_mx_formats_kt1_quasar(
    perf_report,
    format,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    dimensions,
    implied_math_format,
    register_format_hint,
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
    is_perf,
):
    run_matmul(
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        dimensions,
        format,
        implied_math_format,
        register_format_hint,
        enable_direct_indexing,
        transpose,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
