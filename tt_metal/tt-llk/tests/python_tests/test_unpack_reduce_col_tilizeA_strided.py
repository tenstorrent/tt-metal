# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused tilize + column reduce on Wormhole and Blackhole.

Pairs with ``quasar/test_unpack_reduce_col_tilizeA_strided_quasar.py`` and mirrors
``compute_pool_2d.cpp``: a row-major block of ``max_tiles_per_iter`` tiles is tilized by
``_llk_unpack_tilizeA_B_`` against a single reloaded scaler tile, with SrcA cleared to -inf for
Max and to zero (``zero_srcA_reduce``) for Average. Each tile is then column-reduced into its own
dest slot. ``num_faces=2`` is the 16x32 tile pool_2d uses for windows of at most 16 sticks.
"""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceGolden, TilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
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
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

TILE_C_DIM = 32
FACE_R_DIM = 16

UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
]
UNPACK_REDUCE_COL_TILIZEA_STRIDED_POOL_TYPES = [ReducePool.Max, ReducePool.Average]
UNPACK_REDUCE_COL_TILIZEA_STRIDED_NUM_FACES = [2, 4]


def unpack_reduce_col_tilizeA_strided_dest_acc(formats):
    """Float32 input needs a 32-bit dest; bf16 runs both, like large-kernel average pool."""
    if formats.input_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def unpack_reduce_col_tilizeA_strided_max_tiles_per_iter(dest_acc):
    """pool_2d reduces 8 tiles per iteration, or 4 when the block must fit a 32-bit half dest."""
    if dest_acc == DestAccumulation.Yes:
        return [4]
    return [4, 8]


def _math_fidelity(pool_type):
    return MathFidelity.LoFi if pool_type == ReducePool.Max else MathFidelity.HiFi4


@parametrize(
    formats=UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS,
    dest_acc=unpack_reduce_col_tilizeA_strided_dest_acc,
    max_tiles_per_iter=unpack_reduce_col_tilizeA_strided_max_tiles_per_iter,
    num_faces=UNPACK_REDUCE_COL_TILIZEA_STRIDED_NUM_FACES,
    pool_type=UNPACK_REDUCE_COL_TILIZEA_STRIDED_POOL_TYPES,
)
def test_unpack_reduce_col_tilizeA_strided(
    formats,
    dest_acc,
    max_tiles_per_iter,
    num_faces,
    pool_type,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    num_faces_r_dim = num_faces // 2
    tile_dimensions = [FACE_R_DIM * num_faces_r_dim, TILE_C_DIM]
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [tile_dimensions[0], max_tiles_per_iter * TILE_C_DIM]

    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )
    scaler = 1 / tile_dimensions[0] if pool_type == ReducePool.Average else 1
    src_B = torch.full((tile_shape.total_tile_size(),), scaler)

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    templates = [
        MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=pool_type),
        MATH_FIDELITY(_math_fidelity(pool_type)),
    ]
    runtimes = [
        TILE_COUNT(tile_cnt_A),
        NUM_FACES_R_DIM(num_faces_r_dim),
        NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
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
                face_r_dim=FACE_R_DIM,
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
