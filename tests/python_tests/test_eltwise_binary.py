# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    Transpose,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.utils import passed_test


def get_tile_params(tile_dimensions):
    """
    Calculate num_faces and face_r_dim from tile dimensions.

    Supported tile dimensions:
    - [1, 32]  -> face_r_dim=1,  num_faces=2
    - [2, 32]  -> face_r_dim=2,  num_faces=2
    - [4, 32]  -> face_r_dim=4,  num_faces=2
    - [8, 32]  -> face_r_dim=8,  num_faces=2
    - [16, 32] -> face_r_dim=16, num_faces=2
    - [32, 32] -> face_r_dim=16, num_faces=4

    Returns:
        tuple: (num_faces, face_r_dim)
    """
    tile_rows, tile_cols = tile_dimensions

    # face_r_dim is the number of rows per face, capped at 16
    face_r_dim = min(tile_rows, 16)

    # num_faces: 2 for partial tiles (rows < 32), 4 for full 32x32 tiles
    num_faces = (tile_cols // 16) * ((tile_rows + 15) // 16)

    return num_faces, face_r_dim


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    broadcast_type=[BroadcastType.None_],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    transpose_srca=[Transpose.No],
    input_dimensions=[[32, 32], [64, 64]],
    tile_dimensions=[[32, 32]],  # More dimensions coming soon....
)
def test_eltwise_binary(
    formats,
    broadcast_type,
    dest_acc,
    math_fidelity,
    transpose_srca,
    input_dimensions,
    tile_dimensions,
    workers_tensix_coordinates,
):
    num_faces, face_r_dim = get_tile_params(tile_dimensions)

    # Calculate tile count based on tile_dimensions (not hardcoded 32x32)
    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A

    # Generate stimuli has hardcoded tile dims of 32x32
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        # sequential_A=True,
        # const_face=True,
        # const_value_B=2
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        [32, 32],
        BlocksCalculationAlgorithm.Standard,
    )

    # Compute element-wise subtraction in tilized format
    binary_golden = get_golden_generator(EltwiseBinaryGolden)

    golden_tensor = binary_golden(
        MathOperation.Elwsub,
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=MathOperation.Elwsub),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Compare in tilized format
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
