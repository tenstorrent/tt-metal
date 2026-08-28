# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal data-format comparison for Quasar matmul performance.

Every axis except the data format is pinned to a single value, and the input
format always equals the output format, so one row per format is produced and
the only thing that differs between rows is the format itself.

Note on Bfp8_b: Quasar's hardware ``DataFormat`` enum
(``tt_metal/hw/inc/internal/tt-2xx/quasar/tensix_types.h``) has no block-float
family at all, so ``DataFormat::Bfp8_b`` does not exist for ``ARCH_QUASAR`` and
cannot be measured here. MxFp8P (E4M3 plus a shared block exponent) is the
Quasar-generation 8-bit equivalent.
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

# The formats under comparison. Input format == output format for each entry.
COMPARED_FORMATS = [
    DataFormat.Float16_b,
    DataFormat.MxFp8P,
]

# One fixed output block, identical for every format so the cycle counts are
# directly comparable: a dest-full 1x8 tile output accumulated over kt=4.
# DestSync.Half with a 16-bit destination holds 8 tiles, so 1x8 fills it exactly.
MT_DIM, NT_DIM, KT_DIM = 1, 8, 4
DIMENSIONS = [
    [MT_DIM * TILE_DIM, KT_DIM * TILE_DIM],
    [KT_DIM * TILE_DIM, NT_DIM * TILE_DIM],
]


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    format=input_output_formats(COMPARED_FORMATS, same=True),
    # LoFi only: MX inputs are already full precision at LoFi, so the extra HiFi
    # phases would only add rows that are not comparable against the MX ones.
    math_fidelity=[MathFidelity.LoFi],
    dest_sync_mode=[DestSync.Half],
    # A 16-bit destination keeps the tile budget (and therefore DIMENSIONS) the
    # same for both formats.
    dest_acc=[DestAccumulation.No],
    dimensions=[DIMENSIONS],
    # Required for MX: the FPU derives the 2x/MX decode from the src-register
    # format the unpacker sets, not from an explicit ALU override.
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=[None],
    enable_direct_indexing=[False],
    transpose=[Transpose.No],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_matmul_formats_quasar(
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
