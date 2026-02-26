# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    REDUCE_TO_ONE,
    TEST_FACE_DIMS,
)
from helpers.utils import passed_test

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


@parametrize(
    input_dimensions=[[32, 32], [32, 64], [64, 64], [64, 96], [96, 96], [128, 128]],
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    is_reduce_to_one=[False, True],
    math_fidelity=[
        # MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
)
def test_reduce(
    input_dimensions,
    formats,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    workers_tensix_coordinates,
):

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
    )

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:  # result in srcA should be divided by 1
        src_B = torch.full((1024,), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        src_B = torch.full((1024,), 1 / 32)

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=is_reduce_to_one,
    )

    dest_acc = (
        DestAccumulation.Yes
        if (formats.input_format.is_32_bit() or is_dest_acc_needed(formats))
        else DestAccumulation.No
    )
    output_tile_count = 1 if is_reduce_to_one else tile_cnt_A

    DEST_SYNC_TILE_LIMITS = {
        DestSync.Half: 8,
        DestSync.Full: 16,
    }

    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    max_tiles_in_dest = DEST_SYNC_TILE_LIMITS[DestSync.Half] // capacity_divisor

    configuration = TestConfig(
        "sources/reduce_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            TEST_FACE_DIMS(),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(output_tile_count),
            NUM_TILES_IN_BLOCK(max_tiles_in_dest),
            REDUCE_TO_ONE(is_reduce_to_one),
            NUM_FACES(),
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
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
