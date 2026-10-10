# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from conftest import skip_for_quasar, skip_for_wormhole
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
from helpers.utils import passed_test, tolerances

mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


def _fidelities(formats, pool_type):
    """MAX has no fidelity phases; SUM takes the high and one low fidelity (the block math differs between LoFi and HiFi)."""
    if pool_type == ReducePool.Max:
        return [MathFidelity.HiFi4]
    if formats.input_format in [DataFormat.Float16_b, DataFormat.Float32]:
        return [MathFidelity.HiFi2, MathFidelity.HiFi4]
    return [MathFidelity.LoFi, MathFidelity.HiFi4]


def _run_reduce_block(
    formats,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    tile_dimensions,
    input_dimensions,
    reduce_to_one_block,
):
    tile_shape = construct_tile_shape(tile_dimensions)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    if pool_type in [ReducePool.Max, ReducePool.Sum]:
        src_B = torch.full((tile_shape.total_tile_size(),), 1)
    elif reduce_dim == ReduceDimension.Row:
        src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[1])
    elif reduce_dim == ReduceDimension.Column:
        src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[0])
    else:
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
        input_format=formats.input_format,
    )

    dest_acc = (
        DestAccumulation.Yes
        if (
            formats.input_format.is_32_bit()
            or is_dest_acc_needed(formats)
            or (
                formats.output_format == DataFormat.Float32
                and not formats.input_format.is_32_bit()
            )
        )
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
        "sources/reduce_block_test.cpp",
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
            NUM_TILES_IN_BLOCK(
                num_tiles_in_block,
                input_num_tiles_in_block=reduce_to_one_block(tile_cnt_A),
            ),
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

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    if is_reduce_to_one:
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
        ), "Assert against golden failed"
    else:
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
        ), "Assert against golden failed"


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    tile_dimensions=[[32, 32], [16, 32], [32, 16], [16, 16], [1, 32]],
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Sum],
    math_fidelity=_fidelities,
    is_reduce_to_one=[False, True],
)
def test_reduce_block(
    formats,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    tile_dimensions,
):
    """Blocks of a DEST section into their own DEST tiles; reduce to one accumulates 16 tiles in calls of 9 and 7."""
    _run_reduce_block(
        formats,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        tile_dimensions,
        (
            [16 * tile_dimensions[0], tile_dimensions[1]]
            if is_reduce_to_one
            else [256, 32]
        ),
        lambda tile_cnt: 9,
    )


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=[
        *input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
        *input_output_formats([DataFormat.Bfp8_b], same=True),
    ],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Sum],
    call_tiles=[40, 35],
)
def test_reduce_block_chunks(formats, reduce_dim, pool_type, call_tiles):
    """40 full tiles accumulated by block calls: one call of 40 (chunks of 32 and 8), or calls of 35 (32 and 3) and 5."""
    _run_reduce_block(
        formats,
        reduce_dim,
        pool_type,
        MathFidelity.HiFi4,
        True,
        [32, 32],
        [32 * 40, 32],
        lambda tile_cnt: call_tiles,
    )
