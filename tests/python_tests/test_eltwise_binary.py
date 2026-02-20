# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    TransposeGolden,
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
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import SUPPORTED_TILE_SIZES, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

BLACKHOLE_TILE_DIMENSIONS = [[32, 32], [16, 32], [32, 16]]
WORMHOLE_TILE_DIMENSIONS = [list(t) for t in SUPPORTED_TILE_SIZES if t != (16, 16)]


def _get_valid_formats(dest_acc):
    """
    Filter formats based on dest accumulation:
    - If dest accumulation is enabled, input must be Float32
    """
    all_formats = input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b],
        same=False,
    )
    if dest_acc == DestAccumulation.Yes:
        return [f for f in all_formats if f.input_format == DataFormat.Float32]
    return all_formats


def _get_valid_math_fidelity(formats):
    """
    Filter math fidelity based on input data format:
    - Bfp8_b: LoFi only
    - Float16_b: LoFi or HiFi2
    - Float32: HiFi3 and HiFi4
    """
    input_format = formats.input_format
    if input_format == DataFormat.Bfp8_b:
        return [MathFidelity.LoFi]
    elif input_format == DataFormat.Float16_b:
        return [MathFidelity.LoFi, MathFidelity.HiFi2]
    elif input_format == DataFormat.Float32:
        return [MathFidelity.HiFi3, MathFidelity.HiFi4]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def _get_valid_math_ops(math_fidelity):
    """High fidelity operations are only supported for Elwmul."""
    if math_fidelity != MathFidelity.LoFi:
        return [MathOperation.Elwmul]
    return [MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul]


def _get_valid_transpose(broadcast_type):
    """Transpose does not work for Scalar broadcast."""
    if broadcast_type == BroadcastType.Scalar:
        return [Transpose.No]
    return [Transpose.No, Transpose.Yes]


def _get_valid_tile_dimensions(transpose_srca, broadcast_type):
    """
    Filter tile dimensions based on architecture, transpose, and broadcast constraints:
    - Blackhole: only [32,32], [16,32], [32,16]
    - Wormhole: all SUPPORTED_TILE_SIZES except 16x16
    - Transpose only works for 32x32 tiles
    - 32x16 tiles are not supported for Column or Row broadcast
    """
    arch = get_chip_architecture()
    if arch == ChipArchitecture.BLACKHOLE:
        all_tiles = BLACKHOLE_TILE_DIMENSIONS
    else:
        all_tiles = WORMHOLE_TILE_DIMENSIONS

    if transpose_srca == Transpose.Yes:
        return [[32, 32]]

    if broadcast_type in (BroadcastType.Column, BroadcastType.Row):
        return [td for td in all_tiles if td != [32, 16]]

    return all_tiles


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    formats=lambda dest_acc: _get_valid_formats(dest_acc),
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    math_fidelity=lambda formats: _get_valid_math_fidelity(formats),
    transpose_srca=lambda broadcast_type: _get_valid_transpose(broadcast_type),
    math_op=lambda math_fidelity: _get_valid_math_ops(math_fidelity),
    input_dimensions=[[256, 64]],
    tile_dimensions=lambda transpose_srca, broadcast_type: _get_valid_tile_dimensions(
        transpose_srca, broadcast_type
    ),
)
def test_eltwise_binary(
    dest_acc,
    formats,
    broadcast_type,
    math_fidelity,
    transpose_srca,
    math_op,
    input_dimensions,
    tile_dimensions,
    workers_tensix_coordinates,
):

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    # Calculate tile count based on tile_dimensions (not hardcoded 32x32)
    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A

    # Generate stimuli with correct face dimensions for smaller tiles
    # Uses generate_stimuli_w_tile_dimensions which computes face_r_dim and num_faces from tile_dimensions
    src_A, _, src_B, _ = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,  # Use different format for src_B
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    # Compute element-wise subtraction in tilized format
    binary_golden = get_golden_generator(EltwiseBinaryGolden)

    # Tilize inputs for device and golden calculation
    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )

    # Flatten tilized tensors
    src_A_tilized_flat = src_A_tilized.flatten()
    src_B_tilized_flat = src_B_tilized.flatten()

    # Send tilized data to device (device handles transpose during unpack)
    stimuli_A = src_A_tilized_flat
    stimuli_B = src_B_tilized_flat

    # Prepare golden src_A: apply tile-level transpose if enabled
    # Hardware does transpose_faces then transpose_within_faces during unpack
    golden_src_A = src_A_tilized_flat
    if transpose_srca == Transpose.Yes:
        transpose_golden = get_golden_generator(TransposeGolden)
        # Apply face transpose (f0,f1,f2,f3 -> f0,f2,f1,f3)
        golden_src_A = transpose_golden.transpose_faces_multi_tile(
            src_A,
            formats.input_format,
            num_tiles=tile_cnt_A,
            tilize=True,
            untilize=False,  # Keep tilized
            input_dimensions=tuple(input_dimensions),
        )
        # Apply within-face transpose (transpose each 16x16 face)
        golden_src_A = transpose_golden.transpose_within_faces_multi_tile(
            golden_src_A,
            formats.input_format,
            num_tiles=tile_cnt_A,
            tilize=False,  # Already tilized
            untilize=False,  # Keep tilized for golden comparison
            input_dimensions=tuple(input_dimensions),
        )

    # Prepare golden src_B: apply broadcast if enabled
    golden_src_B = src_B_tilized_flat
    if broadcast_type != BroadcastType.None_:
        broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src_B = broadcast_golden(
            broadcast_type,
            src_B_tilized_flat,
            formats.input_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )

    # Compute golden on tilized data
    golden_tensor = binary_golden(
        math_op,
        golden_src_A,
        golden_src_B,
        formats.output_format,
        math_fidelity,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=math_op),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=StimuliConfig(
            stimuli_A,
            formats.input_format,
            stimuli_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
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
