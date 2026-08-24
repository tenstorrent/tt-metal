# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestSync,
    MathFidelity,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    input_output_formats,
    parametrize,
    select_perf_tile_sizes,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tile_shape import construct_tile_shape
from test_eltwise_binary import ALL_TILE_DIMENSIONS, _get_valid_tile_dimensions

_ALL_PERF_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]


def _perf_tile_dimensions(transpose_srca, broadcast_type):
    return select_perf_tile_sizes(
        _get_valid_tile_dimensions(transpose_srca, broadcast_type)
        or ALL_TILE_DIMENSIONS
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]
    ),
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(
        formats, mathop, PERF_RUN=True
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    transpose_srca=[Transpose.No, Transpose.Yes],
    tile_dimensions=lambda transpose_srca, broadcast_type: _perf_tile_dimensions(
        transpose_srca, broadcast_type
    ),
    input_dimensions=lambda dest_acc, tile_dimensions: generate_perf_input_dimensions(
        dest_acc, DestSync.Half, construct_tile_shape(tuple(tile_dimensions))
    ),
)
def test_perf_eltwise_binary(
    perf_report,
    formats,
    mathop,
    math_fidelity,
    dest_acc,
    broadcast_type,
    transpose_srca,
    tile_dimensions,
    input_dimensions,
):
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    if transpose_srca == Transpose.Yes and broadcast_type == BroadcastType.Scalar:
        pytest.skip("SrcA transpose is not supported with scalar broadcast")

    if not tile_dimensions:
        pytest.skip("No perf tile class for this transpose/broadcast combination")

    tile_shape = construct_tile_shape(tuple(tile_dimensions))
    tile_count = (input_dimensions[0] // tile_shape.total_row_dim()) * (
        input_dimensions[1] // tile_shape.total_col_dim()
    )

    # Isolates assume NONE-broadcast dvalid accounting in the dedicated perf kernel.
    if broadcast_type == BroadcastType.None_ and transpose_srca == Transpose.No:
        run_types = _ALL_PERF_RUN_TYPES
    else:
        run_types = [PerfRunType.L1_TO_L1]

    configuration = PerfConfig(
        "sources/eltwise_binary_fpu_perf.cpp",
        formats,
        run_types=run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            BROADCAST_TYPE(broadcast_type),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(8),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
            UNPACK_TRANS_FACES(transpose_srca),
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
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
