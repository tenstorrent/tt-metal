# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHING_FACE,
)
from helpers.tilize_untilize import tilize, tilize_block
from helpers.utils import passed_test


@skip_for_blackhole
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    broadcast_type=[BroadcastType.Column, BroadcastType.Row],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    transpose_srca=[Transpose.Yes],
    input_dimensions=[[32, 32]],
)
def test_eltwise_binary_transpose_bcast(
    formats,
    broadcast_type,
    dest_acc,
    math_fidelity,
    transpose_srca,
    input_dimensions,
    workers_tensix_coordinates,
):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Tilize the input data for hardware
    src_A_tilized = tilize_block(src_A, input_dimensions, formats.input_format)
    src_B_tilized = tilize_block(src_B, input_dimensions, formats.input_format)

    # Compute golden using proper transpose generator that understands tilized data
    transpose_golden = get_golden_generator(TransposeGolden)

    # Apply transpose to srcA: hardware does transpose_faces then transpose_within_faces
    src_A_transposed = transpose_golden.transpose_faces_multi_tile(
        src_A,
        formats.input_format,
        num_tiles=tile_cnt_A,
        tilize=True,  # Tilize before transpose (models hardware behavior)
        input_dimensions=input_dimensions,
    )
    src_A_transposed = transpose_golden.transpose_within_faces_multi_tile(
        src_A_transposed,
        formats.input_format,
        num_tiles=tile_cnt_A,
        untilize=True,  # Untilize after transpose for golden comparison
        input_dimensions=input_dimensions,
    )

    src_B_tilized_for_bcast = tilize(
        src_B, stimuli_format=formats.input_format, num_faces=4
    )
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted_tilized = broadcast_golden(
        broadcast_type,
        src_B_tilized_for_bcast,  # Tilized data
        formats.input_format,
        num_faces=4,
        tile_cnt=tile_cnt_A,
        face_r_dim=16,
    )

    src_A_transposed_tilized = tilize(
        src_A_transposed, stimuli_format=formats.output_format, num_faces=4
    )

    # Compute element-wise subtraction in tilized format
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = binary_golden(
        MathOperation.Elwsub,
        src_A_transposed_tilized,  # Tilized
        src_B_broadcasted_tilized,  # Tilized
        formats.output_format,
        math_fidelity,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_transpose_bcast_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=MathOperation.Elwsub),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHING_FACE(transpose_srca),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Compare in tilized format
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
