# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-operand reduce path (_llk_unpack_reduce_init_ / _llk_unpack_reduce_): the operand
tile goes to SrcA and the scaler tile, placed in its own L1 buffer, to SrcB. The math and
pack threads are the ones test_reduce.py uses, so a mismatch against the golden means the
scaler the unpacker delivered is not the one in the buffer.

REDUCE_ROW SUM/AVG is left out: the math LLK runs that case with the operands swapped
(data in SrcB), which this unpack path does not produce. REDUCE_ROW MAX pools SrcA only.
"""

import math

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
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
from helpers.stimuli_generator import generate_stimuli
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
from helpers.utils import passed_test

pytestmark = [skip_for_quasar]

mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


def _pool_types_for_dim(reduce_dim):
    if reduce_dim == ReduceDimension.Row:
        return [ReducePool.Max]
    return [ReducePool.Max, ReducePool.Average, ReducePool.Sum]


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=_pool_types_for_dim,
    math_fidelity=[MathFidelity.HiFi4],
)
def test_unpack_reduce_scaler(formats, reduce_dim, pool_type, math_fidelity):
    tile_dimensions = [32, 32]
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [64, 32]

    src_A, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    # The scaler tile: 1 for max/sum, 1/N for the average over the reduced extent.
    if pool_type in [ReducePool.Max, ReducePool.Sum]:
        scaler = 1.0
    elif reduce_dim == ReduceDimension.Row:
        scaler = 1 / tile_dimensions[1]
    elif reduce_dim == ReduceDimension.Column:
        scaler = 1 / tile_dimensions[0]
    else:
        scaler = 1 / math.sqrt(tile_dimensions[0] * tile_dimensions[1])
    src_B = torch.full((tile_shape.total_tile_size(),), scaler)

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=False,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

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
        "sources/unpack_reduce_scaler_test.cpp",
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
            OUTPUT_TILE_CNT(tile_cnt_A),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            REDUCE_TO_ONE(False),
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
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Print the first reduced row so a scaler mismatch is visible in the log.
    print(
        f"device[:8]={res_tensor[:8].tolist()} golden[:8]={golden_tensor[:8].tolist()}"
    )

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
