# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Float16_b vs MxFp8P matmul performance with a single K tile.

Same comparison as perf_matmul_formats_quasar.py, but with kt=1 instead of kt=4:
the output block stays a dest-full 1x8 tiles, so the operand matrices shrink from
[32,128] x [128,256] (36 input tiles) to [32,32] x [32,256] (9 input tiles).

Two consequences worth knowing when reading the report:

* A step of the TILE_LOOP normalisation still covers 32x32x32 = 32768 MAC, so the
  per-step figures stay directly comparable to the kt=4 report.
* ``tile_cnt`` drops from 32 to 8, so each step now carries one whole packed
  output tile instead of a quarter of one. Packing cost per step therefore rises
  roughly fourfold (measured 8.10 -> 32.44 cycles for Float16_b), which moves the
  bottleneck: at kt=4 unpack alone sets the limit, at kt=1 unpack and pack are
  both near it and contend.

Kept in its own module so its report lands in perf_data/perf_matmul_formats_kt1_quasar/
and does not overwrite the kt=4 results.
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

COMPARED_FORMATS = [
    DataFormat.Float16_b,
    DataFormat.MxFp8P,
]

# Dest-full 1x8 output tiles, accumulated over a single K tile.
MT_DIM, NT_DIM, KT_DIM = 1, 8, 1
DIMENSIONS = [
    [MT_DIM * TILE_DIM, KT_DIM * TILE_DIM],
    [KT_DIM * TILE_DIM, NT_DIM * TILE_DIM],
]


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=input_output_formats(COMPARED_FORMATS, same=True),
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=[DestSync.Half],
    dest_acc=[DestAccumulation.No],
    dimensions=[DIMENSIONS],
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=[None],
    enable_direct_indexing=[False],
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_formats_kt1_quasar(
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
