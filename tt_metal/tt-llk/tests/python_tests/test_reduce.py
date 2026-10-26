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
    PerfRunType,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    IN_FACE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    REDUCE_TO_ONE,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import (
    passed_test,
    skip_ttsim_partial_face_bfp_zero_write,
    tolerances,
)

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}

REDUCE_DIMS = [ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar]
POOL_TYPES = [ReducePool.Max, ReducePool.Average, ReducePool.Sum]
ALL_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]
REDUCE_TILE_DIMENSIONS = [
    [1, 32],
    [2, 32],
    [4, 32],
    [8, 32],
    [16, 16],
    [16, 32],
    [32, 32],
    [32, 16],
]


def _fidelities_for_format(formats):
    """LoFi fails for Float16_b/Float32 inputs — exclude at parametrize time."""
    if formats.input_format in [DataFormat.Float16_b, DataFormat.Float32]:
        return [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]
    return ALL_FIDELITIES


def _reduce_to_one_for_format(formats):
    """Bfp8_b accumulates error easily with reduce_to_one (#45143) — exclude at parametrize time."""
    if (
        formats.input_format == DataFormat.Bfp8_b
        or formats.output_format == DataFormat.Bfp8_b
    ):
        return [False]
    return [False, True]


def _dest_acc_required(formats):
    # Float32 golden uses full FP32 accumulation; match that in HW whenever we
    # pack Float32 from sub-32-bit float inputs (e.g. Float16_b), otherwise
    # HiFi column/reduce-to-one cases can drift below PCC thresholds.
    return (
        formats.input_format.is_32_bit()
        or is_dest_acc_needed(formats)
        or (
            formats.output_format == DataFormat.Float32
            and not formats.input_format.is_32_bit()
        )
    )


def reduce_dest_acc_modes(formats):
    if _dest_acc_required(formats):
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def _required_dest_acc_mode(formats):
    return [
        DestAccumulation.Yes if _dest_acc_required(formats) else DestAccumulation.No
    ]


REDUCE_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)

REDUCE_SWEEP = dict(
    tile_dimensions=REDUCE_TILE_DIMENSIONS,
    formats=REDUCE_FORMATS,
    dest_acc=reduce_dest_acc_modes,
    reduce_dim=REDUCE_DIMS,
    pool_type=POOL_TYPES,
    math_fidelity=_fidelities_for_format,
    is_reduce_to_one=_reduce_to_one_for_format,
)


def _scaler_tile(tile_shape, tile_dimensions, reduce_dim, pool_type):
    if pool_type in [ReducePool.Max, ReducePool.Sum]:
        # result in srcA should be divided by 1
        return torch.full((tile_shape.total_tile_size(),), 1)
    # reduce average divides by length of elements in array we reduce
    if reduce_dim == ReduceDimension.Row:
        return torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[1])
    if reduce_dim == ReduceDimension.Column:
        return torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[0])
    return torch.full(
        (tile_shape.total_tile_size(),),
        1 / math.sqrt(tile_dimensions[0] * tile_dimensions[1]),
    )


def _functional_input_dimensions(is_reduce_to_one):
    # Accumulating large tensors into a single value can lead to significant numerical errors
    # especially if the sum/average of the dimension is not enough to increment to the next representable value in the output format.
    # To mitigate this, we use smaller input values which reduces the chance of large accumulation errors.
    return [128, 32] if is_reduce_to_one else [256, 32]


def _run_reduce_test(
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    tile_dimensions,
    *,
    input_dimensions=None,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    tile_shape = construct_tile_shape(tile_dimensions)
    skip_ttsim_partial_face_bfp_zero_write(
        tile_shape, formats.output_format, reduce_dim
    )
    if input_dimensions is None:
        input_dimensions = _functional_input_dimensions(is_reduce_to_one)

    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )
    # One resident scaler tile, as in production.
    src_B = _scaler_tile(tile_shape, tile_dimensions, reduce_dim, pool_type)

    output_tile_count = 1 if is_reduce_to_one else tile_cnt_A

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/reduce_test.cpp",
        "formats": formats,
        "templates": [
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
        ],
        "runtimes": [
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            TILE_COUNT(tile_cnt_A),
            OUTPUT_TILE_CNT(output_tile_count),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            REDUCE_TO_ONE(is_reduce_to_one),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=1,
            tile_count_res=output_tile_count,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types or [PerfRunType.L1_TO_L1],
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

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

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

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
        ), "Assert against golden failed"
    else:
        # Use default target_pcc = 0.99
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
        ), "Assert against golden failed"


@parametrize(**REDUCE_SWEEP)
def test_reduce(
    tile_dimensions,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
):
    _run_reduce_test(
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        tile_dimensions,
    )


@parametrize(
    formats=[
        fmt
        for fmt in input_output_formats(
            [
                DataFormat.Bfp4_b,
                DataFormat.Float16_b,
                DataFormat.Bfp8_b,
                DataFormat.Float32,
            ]
        )
        if fmt.input_format == DataFormat.Bfp4_b
        or fmt.output_format == DataFormat.Bfp4_b
    ],
    dest_acc=_required_dest_acc_mode,
    reduce_dim=REDUCE_DIMS,
    pool_type=POOL_TYPES,
    math_fidelity=ALL_FIDELITIES,
    is_reduce_to_one=[False, True],
    tile_dimensions=[[32, 32]],
)
def test_reduce_bfp4_b(
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    tile_dimensions,
):
    _run_reduce_test(
        formats,
        dest_acc,
        reduce_dim,
        pool_type,
        math_fidelity,
        is_reduce_to_one,
        tile_dimensions,
    )
