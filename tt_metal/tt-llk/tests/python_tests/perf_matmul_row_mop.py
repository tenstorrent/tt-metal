# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.llk_params import DestAccumulation, MathFidelity, PerfRunType, Transpose
from helpers.matmul_sweep import (
    DEST_HALF_BFP_PACK_HANG_REASON,
    PERF_RING_TILES,
    generate_tile_dims,
    is_dest_half_bfp_pack_hang,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_ROW_MOP,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from perf_matmul import matmul_combos

# The blocks the bmm kernel runs on the row MOP: eight tiles or more, more than one k step, and not a Float32 output packed
# from a 16-bit DEST; perf_matmul has the same rows on the tile MOP.
ROW_MOP_MIN_TILES = 8


def _row_mop_combos():
    combos = matmul_combos(
        formats=input_output_formats(
            [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b]
        ),
        dest_acc=lambda formats: (
            [DestAccumulation.Yes]
            if is_dest_acc_needed(formats)
            else [DestAccumulation.No, DestAccumulation.Yes]
        ),
    )
    out = []
    for combo in combos:
        formats, dest_acc, dims = combo[0], combo[1], generate_tile_dims(combo[3])
        float32_from_16bit_dest = (
            formats.output_format == DataFormat.Float32
            and dest_acc == DestAccumulation.No
        )
        if (
            dims.rt_dim * dims.ct_dim >= ROW_MOP_MIN_TILES
            and dims.kt_dim > 1
            and not float32_from_16bit_dest
        ):
            out.append(combo)
    return out


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    combos=_row_mop_combos(),
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_perf_matmul_row_mop(perf_report, combos, math_fidelity):

    formats, dest_acc, dest_sync, (matrix_a, matrix_b) = combos

    if is_dest_half_bfp_pack_hang(dest_sync, dest_acc, formats):
        pytest.skip(DEST_HALF_BFP_PACK_HANG_REASON)

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    dims = generate_tile_dims((matrix_a, matrix_b))
    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)

    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            MATMUL_ROW_MOP(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
