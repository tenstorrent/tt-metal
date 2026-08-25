# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

# Cold start, functional-max inner dim, and reuse/bandwidth.
KT_DIMS = [1, 4, 32]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]


def dest_corner_mn(max_tiles: int) -> List[tuple]:
    """1×1, 1×max, max×1, and the largest square that fits in dest."""
    square = int(max_tiles**0.5)
    corners = [(1, 1), (1, max_tiles), (max_tiles, 1), (square, square)]
    return list(dict.fromkeys(corners))


def generate_dest_corner_combinations(max_tiles: int, kt_dims=KT_DIMS) -> List[tuple]:
    return [
        (
            [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
            [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
        )
        for mt_dim, nt_dim in dest_corner_mn(max_tiles)
        for kt_dim in kt_dims
    ]


def matmul_combos(
    formats: List[FormatConfig],
    dest_acc: List[DestAccumulation],
):
    def _dest_bank_max_tiles(
        format: FormatConfig, dest_acc: DestAccumulation, dest_sync: DestSync
    ):
        capacity_divisor = (
            2 if is_dest_acc_needed(format) or dest_acc == DestAccumulation.Yes else 1
        )
        return DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor

    unique_max_tiles = set(
        _dest_bank_max_tiles(fmt, acc, sync)
        for fmt in formats
        for acc in dest_acc
        for sync in DEST_SYNC_MODES
        if not (fmt.input_format.is_32_bit() and acc == DestAccumulation.No)
    )
    dimensions = {
        max_tiles: generate_dest_corner_combinations(max_tiles, kt_dims=KT_DIMS)
        for max_tiles in unique_max_tiles
    }

    return [
        (format, accumulation, dest_sync, dims)
        for format in formats
        for accumulation in dest_acc
        for dest_sync in DEST_SYNC_MODES
        if not (format.input_format.is_32_bit() and accumulation == DestAccumulation.No)
        for dims in dimensions[_dest_bank_max_tiles(format, accumulation, dest_sync)]
    ]


@pytest.mark.perf
@parametrize(
    combos=matmul_combos(
        formats=input_output_formats(
            [
                DataFormat.Float16_b,
                DataFormat.Float16,
                DataFormat.Float32,
                DataFormat.Bfp8_b,
            ]
        ),
        dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    ),
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_perf_matmul(
    perf_report,
    combos,
    math_fidelity,
):

    formats, dest_acc, dest_sync, (matrix_a, matrix_b) = combos

    if is_dest_acc_needed(formats) and dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("32-bit inputs require dest accumulation")

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    # Calculate all matmul dimensions using helper function
    dims = generate_tile_dims((matrix_a, matrix_b))

    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim

    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(dims.rt_dim * dims.ct_dim),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=variant_tile_count,
            tile_count_B=variant_tile_count,
            tile_count_res=dims.rt_dim * dims.ct_dim,
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
