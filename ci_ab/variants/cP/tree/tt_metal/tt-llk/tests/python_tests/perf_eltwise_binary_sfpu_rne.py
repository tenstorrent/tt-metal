# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Binary SFPU ADD/SUB/RSUB with DstRoundingMode::NearestEven.

This is the arm binary_ng runs for every bf16 ADD/SUB/RSUB it sends to the SFPU: the
fp32 result is narrowed into the bf16 Dest with a software round-to-nearest-even. It
is only meaningful with a bf16 Dest (dest_acc=No); with fp32 accumulation the rounding
is skipped and the kernel is the Default one perf_eltwise_binary_sfpu already measures.

It lives in its own module, and so in its own perf catalog entry, because the perf
gate keys a point on its non-null sweep columns: recording ``dst_rounding`` in
perf_eltwise_binary_sfpu would re-key every point of that module and detach it from
its baseline.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    DstRoundingMode,
    MathOperation,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BROADCAST_TYPE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    SFPU_BCAST_DIM,
    SFPU_DST_ROUNDING_MODE,
    TILE_COUNT,
    generate_input_dim,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwrsub,
    ],
    loop_factor=[16],
    iterations=[32],
    input_dimensions=[[256, 128]],
)
def test_perf_eltwise_binary_sfpu_rne(
    perf_report,
    formats,
    mathop,
    loop_factor,
    iterations,
    input_dimensions,
):
    # The same kernel and blocks as perf_eltwise_binary_sfpu's bf16 rows, in NearestEven mode.
    dest_acc = DestAccumulation.No
    tile_count, _, _ = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
    )

    configuration = PerfConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(iterations),
            BROADCAST_TYPE(LlkBroadcastType.None_),
            SFPU_BCAST_DIM(LlkBroadcastType.None_),
            SFPU_DST_ROUNDING_MODE(DstRoundingMode.NearestEven),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
