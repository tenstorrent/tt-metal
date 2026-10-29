# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused tilize plus column reduce on Wormhole and Blackhole."""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceGolden, TilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, MAX_FACE_R_DIM
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
]
UNPACK_REDUCE_COL_TILIZEA_STRIDED_POOL_TYPES = [ReducePool.Max, ReducePool.Average]
UNPACK_REDUCE_COL_TILIZEA_STRIDED_DEST_SYNC = [DestSync.Half, DestSync.Full]
UNPACK_REDUCE_COL_TILIZEA_STRIDED_NUM_FACES = [2, 4]
# num_faces=4 is a full 32-row tile. [16, 32] is one face-row, so partial heights stay on num_faces=2.
UNPACK_REDUCE_COL_TILIZEA_STRIDED_PARTIAL_FACE_R_DIMS = [8, 4]


def unpack_reduce_col_tilizeA_strided_dest_acc(formats):
    """Sweep Float32 input only with a 32-bit dest. That is a coverage choice; bf16 runs both modes."""
    if formats.input_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def unpack_reduce_col_tilizeA_strided_max_tiles_per_iter(dest_acc):
    """pool_2d reduces 8 tiles per iteration, or 4 when the block must fit a 32-bit half dest."""
    if dest_acc == DestAccumulation.Yes:
        return [4]
    return [4, 8]


def unpack_reduce_col_tilizeA_strided_face_r_dim(num_faces):
    if num_faces == 2:
        return [MAX_FACE_R_DIM, *UNPACK_REDUCE_COL_TILIZEA_STRIDED_PARTIAL_FACE_R_DIMS]
    return [MAX_FACE_R_DIM]


def _math_fidelity(pool_type):
    return MathFidelity.LoFi if pool_type == ReducePool.Max else MathFidelity.HiFi4


@parametrize(
    formats=UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS,
    dest_acc=unpack_reduce_col_tilizeA_strided_dest_acc,
    max_tiles_per_iter=unpack_reduce_col_tilizeA_strided_max_tiles_per_iter,
    num_faces=UNPACK_REDUCE_COL_TILIZEA_STRIDED_NUM_FACES,
    face_r_dim=unpack_reduce_col_tilizeA_strided_face_r_dim,
    pool_type=UNPACK_REDUCE_COL_TILIZEA_STRIDED_POOL_TYPES,
    dest_sync=UNPACK_REDUCE_COL_TILIZEA_STRIDED_DEST_SYNC,
)
def test_unpack_reduce_col_tilizeA_strided(
    formats,
    dest_acc,
    max_tiles_per_iter,
    num_faces,
    pool_type,
    face_r_dim,
    dest_sync,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    num_faces_r_dim = num_faces // 2
    tile_dimensions = [face_r_dim * num_faces_r_dim, DEFAULT_TILE_C_DIM]
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [tile_dimensions[0], max_tiles_per_iter * DEFAULT_TILE_C_DIM]

    # Negative Max values lose to a zeroed unwritten row, so a skipped -inf clear fails.
    spec_A = (
        StimuliSpec.uniform(low=-1.1, high=-0.1)
        if pool_type == ReducePool.Max
        else None
    )
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        spec_A=spec_A,
        tile_dimensions=tile_dimensions,
    )
    # Only face 0 holds the scaler. Average uses SrcB, so a missed reload scales later faces by 0.
    scaler = 1 / tile_dimensions[0] if pool_type == ReducePool.Average else 1
    src_B = torch.zeros(tile_shape.total_tile_size())
    src_B[: tile_shape.face_r_dim * tile_shape.face_c_dim] = scaler

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    templates = [
        MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=pool_type),
        MATH_FIDELITY(_math_fidelity(pool_type)),
        DEST_SYNC(dest_sync),
    ]
    runtimes = [
        TILE_COUNT(tile_cnt_A),
        NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
        NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
        TEST_FACE_DIMS(face_r_dim=face_r_dim),
        LOOP_FACTOR(loop_factor),
    ]
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types or [PerfRunType.L1_TO_L1],
        test_config_kwargs={
            "test_name": "sources/unpack_reduce_col_tilizeA_strided_test.cpp",
            "formats": formats,
            "templates": templates,
            "runtimes": runtimes,
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=1,
                tile_count_res=tile_cnt_A,
                num_faces=num_faces,
                face_r_dim=face_r_dim,
                tile_dimensions=tile_dimensions,
                use_dense_tile_dimensions=True,
            ),
            "unpack_to_dest": False,
            "dest_acc": dest_acc,
        },
    )
    if is_perf:
        configuration.run(perf_report)
        return

    # The device reads operand A row-major, so the golden reduces the tilized block.
    golden_A = get_golden_generator(TilizeGolden)(
        src_A,
        input_dimensions,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
    )
    golden_tensor = get_golden_generator(ReduceGolden)(
        golden_A,
        ReduceDimension.Column,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, tile_shape=tile_shape
    ), "Assert against golden failed"
