# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Perf counterpart of test_bcast.py for the 32-bit unpack-to-dest broadcast datacopy.

The 32-bit path in ``_llk_math_eltwise_unary_datacopy_`` (``unpack_to_dest &&
is_32bit_input(...)``) issues an explicit MOVD2B/MOVB2D stream rather than the preconfigured
MOP, so its cost is a math-thread instruction-issue cost. MATH_ISOLATE is the metric that reads
it; L1_TO_L1 gives the end-to-end number. Every other perf module pins BROADCAST_TYPE to NONE
and so never enters this branch.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


@pytest.mark.perf
@parametrize(
    # Only the 32-bit formats reach the measured branch: is_32bit_input() gates on
    # Float32/Int32 in and out, and dest_acc must be Yes for unpack-to-dest.
    formats=input_output_formats([DataFormat.Float32, DataFormat.Int32], same=True),
    broadcast_type=[
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ],
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    input_dimensions=[
        [128, 64],  # tile_cnt: 8, i.e. two full 4-tile 32-bit dest blocks
    ],
)
def test_perf_bcast(
    perf_report,
    formats,
    broadcast_type,
    loop_factor,
    input_dimensions,
):
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/unpack_a_bcast_datacopy_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
            PerfRunType.L1_TO_L1,
        ],
        templates=[
            BROADCAST_TYPE(broadcast_type),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )

    configuration.run(perf_report)
