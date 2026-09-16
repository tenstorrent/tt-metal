# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Tuple

import pytest
import torch
from helpers.constraints import (
    get_valid_data_format_conversions,
    get_valid_dest_accumulation_modes,
)
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    ReduceGolden,
    TilizeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    generate_perf_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import is_mx_unsupported_tile_dims
from helpers.tile_shape import TileShape, construct_tile_shape
from helpers.utils import passed_test

# 32x32 uses the four face MOP; the Nx32 tiles (N <= 8) use the tiny tile MOP in
# llk_unpack_reduce_col_tilizeA_strided.h, where one strided unpack covers a whole face.
# Tile shapes with a full 16 row face and fewer than four faces (16x16, 16x32, 32x16) have
# no MOP on this path yet.
UNPACK_REDUCE_COL_TILIZEA_STRIDED_TILE_SIZES = [
    (32, 32),
    (1, 32),
    (2, 32),
    (4, 32),
    (8, 32),
]


def generate_corner_case_input_dimensions(
    dest_sync: DestSync, dest_acc: DestAccumulation, tile_shape: TileShape
) -> List[List[int]]:
    """
    Generate input dimensions covering the corner cases of the unpack loop nest.

    The tile counts are 1 tile (minimum), max-wide (stresses block_ct), max-tall (stresses
    block_rt) and max-square (both loops at capacity), where max is the number of tiles that
    fit into dest for the given dest_sync / dest_acc pair.
    """
    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    max_tiles_in_dest = DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor
    square_side = math.isqrt(max_tiles_in_dest)

    tile_counts = [
        (1, 1),
        (1, max_tiles_in_dest),
        (max_tiles_in_dest, 1),
        (square_side, max_tiles_in_dest // square_side),
    ]

    return [
        [rt_dim * tile_shape.total_row_dim(), ct_dim * tile_shape.total_col_dim()]
        for rt_dim, ct_dim in tile_counts
    ]


def generate_unpack_reduce_col_tilizeA_strided_combinations(
    formats_list: List[FormatConfig],
    *,
    is_perf=False,
):
    """
    Generate unpack_reduce_col_tilizeA_strided test combinations for Quasar.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc, dest_sync, input_dimensions, pool_type, tile_dimensions) tuples
    """

    def _requires_dest_acc_for_reduce(in_fmt, out_fmt):
        """Int8->Int8 and UInt8->UInt8 reduce ops need 32-bit dest.
        This is in addition to the base constraints which are true for every operation.
        """
        return in_fmt in (DataFormat.Int8, DataFormat.UInt8) and in_fmt == out_fmt

    tile_sizes = (
        select_perf_tile_sizes(UNPACK_REDUCE_COL_TILIZEA_STRIDED_TILE_SIZES)
        if is_perf
        else UNPACK_REDUCE_COL_TILIZEA_STRIDED_TILE_SIZES
    )

    combinations = []

    for fmt in get_valid_data_format_conversions(formats_list):
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        # Unpack to dest is not supported for unpack tilize operands, so the input cannot be Int32
        if in_fmt == DataFormat.Int32:
            continue
        for acc in get_valid_dest_accumulation_modes(fmt):
            if (
                _requires_dest_acc_for_reduce(in_fmt, out_fmt)
                and acc == DestAccumulation.No
            ):
                continue
            for dest_sync in (
                (DestSync.Half,) if is_perf else (DestSync.Half, DestSync.Full)
            ):
                for tile_dimensions in tile_sizes:
                    if is_mx_unsupported_tile_dims(in_fmt, out_fmt, tile_dimensions):
                        continue
                    tile_shape = construct_tile_shape(tile_dimensions)
                    dimensions_list = (
                        generate_perf_input_dimensions(acc, dest_sync, tile_shape)
                        if is_perf
                        else generate_corner_case_input_dimensions(
                            dest_sync, acc, tile_shape
                        )
                    )
                    for dimensions in dimensions_list:
                        for pool_type in (
                            ReducePool.Max,
                            ReducePool.Sum,
                            ReducePool.Average,
                        ):
                            if pool_type == ReducePool.Average and in_fmt.is_integer():
                                continue
                            combinations.append(
                                (
                                    fmt,
                                    acc,
                                    dest_sync,
                                    dimensions,
                                    pool_type,
                                    runtime(tile_dimensions),
                                )
                            )

    return combinations


UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.Int32,
    ],
)
ALL_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS = (
    generate_unpack_reduce_col_tilizeA_strided_combinations(
        UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS
    )
)
PERF_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS = (
    generate_unpack_reduce_col_tilizeA_strided_combinations(
        UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS,
        is_perf=True,
    )
)


def unpack_reduce_col_tilizeA_strided_implied_math_formats():
    return [ImpliedMathFormat.No]


def generate_reduce_scaler(
    pool_type: ReducePool, tile_shape: TileShape, tile_dimensions: Tuple[int, int]
) -> torch.Tensor:
    """SrcB scaler tile: 1.0 per element, or 1/rows for Average so the column reduce gives the mean."""
    scaler = 1 / tile_dimensions[0] if pool_type == ReducePool.Average else 1
    return torch.full((tile_shape.total_tile_size(),), scaler)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims=ALL_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
    implied_math_format=unpack_reduce_col_tilizeA_strided_implied_math_formats,
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unpack_reduce_col_tilizeA_strided_quasar(
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims,
    implied_math_format,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    (
        formats,
        dest_acc,
        dest_sync_mode,
        input_dimensions,
        pool_type,
        tile_dimensions,
    ) = formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims

    tile_shape = construct_tile_shape(tile_dimensions)
    num_faces = tile_shape.total_num_faces()
    reduce_dim = ReduceDimension.Column
    math_fidelity = MathFidelity.LoFi

    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    src_B = generate_reduce_scaler(pool_type, tile_shape, tile_dimensions)

    tilize_gen = get_golden_generator(TilizeGolden)

    golden_src_A = src_A
    input_fmt = formats.input_format

    if formats.input_format.is_mx_format():
        golden_src_A = quantize_mx_tensor_chunked(src_A, formats.input_format)
        input_fmt = DataFormat.Float16_b

    golden_A = tilize_gen(
        golden_src_A,
        input_dimensions,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
    )

    if not is_perf:
        reduce_gen = get_golden_generator(ReduceGolden)
        golden_tensor = reduce_gen(
            golden_A,
            reduce_dim,
            pool_type,
            formats.output_format,
            tile_cnt_A,
            tile_shape=tile_shape,
            input_format=input_fmt,
        )

    mathop = {
        ReduceDimension.Row: MathOperation.ReduceRow,
        ReduceDimension.Column: MathOperation.ReduceColumn,
        ReduceDimension.Scalar: MathOperation.ReduceScalar,
    }[reduce_dim]

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/unpack_reduce_col_tilizeA_strided_quasar_test.cpp",
        "formats": formats,
        "templates": [
            IMPLIED_MATH_FORMAT(implied_math_format),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=tile_dimensions
            ),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_FACES(num_faces),
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
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
        boot_mode=boot_mode,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, tile_shape=tile_shape
    ), "Assert against golden failed"
