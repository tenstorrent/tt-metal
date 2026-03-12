# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathFidelity,
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
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    IN_FACE_DIMS,
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    REDUCE_TO_ONE,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, tolerances

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


@parametrize(
    tile_dimensions=[[1, 32], [2, 32], [4, 32], [8, 32], [16, 32], [32, 32], [32, 16]],
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    is_reduce_to_one=[False, True],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
    math_fidelity=[
        # MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_reduce(
    formats,
    reduce_dim,
    pool_type,
    is_reduce_to_one,
    math_fidelity,
    workers_tensix_coordinates,
    tile_dimensions,
):

    tile_shape = construct_tile_shape(tile_dimensions)

    if is_reduce_to_one:
        # Accumulating large tensors into a single value can lead to significant numerical errors
        # especially if the sum/average of the dimension is not enough to increment to the next representable value in the output format.
        # To mitigate this, we use smaller input values which reduces the chance of large accumulation errors.
        input_dimensions = [128, 32]
    else:
        # If not reducing to one, we can use larger input dimensions to better test the reduction operation without excessive numerical errors in the accumulation.
        input_dimensions = [256, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:  # result in srcA should be divided by 1
        src_B = torch.full((tile_shape.total_tile_size(),), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        if reduce_dim == ReduceDimension.Row:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[1])
        elif reduce_dim == ReduceDimension.Column:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[0])
        else:  # Scalar
            src_B = torch.full(
                (tile_shape.total_tile_size(),),
                1 / math.sqrt(tile_dimensions[0] * tile_dimensions[1]),
            )

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=is_reduce_to_one,
        tile_shape=tile_shape,
    )

    dest_acc = (
        DestAccumulation.Yes
        if (formats.input_format.is_32_bit() or is_dest_acc_needed(formats))
        else DestAccumulation.No
    )

    output_tile_count = 1 if is_reduce_to_one else tile_cnt_A

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/reduce_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(output_tile_count),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            REDUCE_TO_ONE(is_reduce_to_one),
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
            tile_count_res=output_tile_count,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    if is_reduce_to_one:
        # Lower the threshold for reduce to one cases as they are more prone to numerical errors, especially in lower precision formats
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
            custom_pcc_threshold=(
                0.90
                if formats.output_format is not DataFormat.Bfp8_b
                else pow(0.99, tile_cnt_A)
            ),
            custom_atol=tolerances[formats.output_format].atol * tile_cnt_A,
            custom_rtol=tolerances[formats.output_format].rtol * tile_cnt_A,
            print_errors=True,
        ), "Assert against golden failed"
    else:
        # Use default target_pcc = 0.99
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
            print_errors=True,
        ), "Assert against golden failed"
