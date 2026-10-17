# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import quantize_input_to_unpack_format
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathOperation,
    ReduceDimension,
    ReducePool,
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
    IN_FACE_DIMS,
    INPUT_TILE_CNT,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


def _kept_datums(tile_shape, reduce_dim):
    """Datums the reduce mask keeps, in the face-major order the tile is packed in."""
    kept = torch.zeros(
        tile_shape.total_num_faces(),
        tile_shape.face_r_dim,
        tile_shape.face_c_dim,
        dtype=torch.bool,
    )
    for face in range(tile_shape.total_num_faces()):
        face_row, face_col = divmod(face, tile_shape.num_faces_c_dim)
        if reduce_dim == ReduceDimension.Row and face_col == 0:
            kept[face, :, 0] = True
        elif reduce_dim == ReduceDimension.Column and face_row == 0:
            kept[face, 0, :] = True
    if reduce_dim == ReduceDimension.Scalar:
        kept[0, 0, 0] = True
    return kept.flatten()


@parametrize(
    tile_dimensions=[
        [1, 32],
        [2, 32],
        [4, 32],
        [8, 32],
        [16, 16],
        [16, 32],
        [32, 32],
        [32, 16],
    ],
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b], same=True
    ),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
)
def test_pack_reduce_mask(formats, reduce_dim, tile_dimensions):
    """Pack a datacopied tile through the reduce mask and check the padding is zero.

    DEST holds the full input tile rather than a reduction result, so any datum the mask
    fails to drop reaches L1 instead of hiding behind a zero that the reduction left in DEST.
    """
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [4 * tile_dimensions[0], tile_dimensions[1]]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    torch_format = format_dict[formats.output_format]
    kept = _kept_datums(tile_shape, reduce_dim).repeat(tile_cnt_A)
    golden_tensor = (
        quantize_input_to_unpack_format(src_A, formats.input_format)
        .to(torch_format)
        .clone()
    )
    golden_tensor[~kept] = 0

    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/pack_reduce_mask_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=ReducePool.Sum),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
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
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    leaked = torch.count_nonzero(res_tensor[~kept]).item()
    assert leaked == 0, f"{leaked} datums outside the reduce mask were packed"
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, tile_shape=tile_shape
    )
