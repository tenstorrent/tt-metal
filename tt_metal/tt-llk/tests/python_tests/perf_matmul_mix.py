# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import DestAccumulation, MathFidelity, PerfRunType, Transpose
from helpers.matmul_sweep import PERF_RING_TILES, generate_tile_dims, matmul_pack_form
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_PACK_BLOCK,
    MATMUL_ROW_MOP,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    MATMUL_UNPACK_TTSYNC,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from perf_matmul import matmul_combos


# SrcA and SrcB formats that differ (perf_matmul and perf_math_matmul sweep equal pairs only): Bfp8_b weights against Float16_b or Float32
# activations in both orders, with a Float32 output. The harness convention: input_format is SrcA (matmul in1), input_format_B is SrcB (in0).
MIXED_FORMATS = [
    InputOutputFormat(
        DataFormat.Bfp8_b, DataFormat.Float32, input_format_B=DataFormat.Float16_b
    ),
    InputOutputFormat(
        DataFormat.Float16_b, DataFormat.Float32, input_format_B=DataFormat.Bfp8_b
    ),
    InputOutputFormat(
        DataFormat.Bfp8_b, DataFormat.Float32, input_format_B=DataFormat.Float32
    ),
    InputOutputFormat(
        DataFormat.Float32, DataFormat.Float32, input_format_B=DataFormat.Bfp8_b
    ),
]


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    combos=matmul_combos(
        formats=MIXED_FORMATS,
        dest_acc=lambda formats: (
            [DestAccumulation.Yes]
            if is_dest_acc_needed(formats)
            else [DestAccumulation.No, DestAccumulation.Yes]
        ),
    ),
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_perf_matmul_mix(
    perf_report,
    combos,
    math_fidelity,
):

    formats, dest_acc, dest_sync, (matrix_a, matrix_b) = combos

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
    # PERF_ADDRESS rings are PERF_RING_TILES (tests/helpers/include/perf.h).
    # Cap L1 stimuli so K=32 dest-fill Float32 does not overflow Tensix L1.
    # Unpack MOP occupancy is filtered in generate_dest_corner_combinations.
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)


    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, dest_acc, 0),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            *MATMUL_STREAM_NARROW.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(formats, math_fidelity, dims.ct_dim, dims.rt_dim, dest_sync),
            *MATMUL_STREAM_Z.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
                kt_dim=dims.kt_dim,
            ),
            *MATMUL_UNPACK_TTSYNC.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim, dest_sync
            ),
            *MATMUL_K_LOOP.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim
            ),
            *MATMUL_ROW_END_FOLD.for_block(dims.ct_dim, dims.rt_dim),
            *MATMUL_ROW_MOP.for_block(
                formats, dest_acc, math_fidelity, dest_sync, dims.ct_dim, dims.rt_dim, dims.kt_dim
            ),
            *MATMUL_FORM_KEYS.for_block(
                formats, dims.kt_dim, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity, dest_sync=dest_sync, dest_acc=dest_acc, unfuse_one_tile=True,
            ),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    dest_sync,
                    math_fidelity,
                    dims.rt_dim,
                    dims.ct_dim,
                    dims.kt_dim,
                )
            ),
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
            formats.input_format_B,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
