# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

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
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test


def generate_unpack_reduce_col_tilizeA_strided_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_reduce_col_tilizeA_strided test combinations for Quasar.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc, dest_sync, input_dimensions, pool_type) tuples
    """

    def _requires_dest_acc_for_reduce(in_fmt, out_fmt):
        """Int8->Int8 and UInt8->UInt8 reduce ops need 32-bit dest.
        This is in addition to the base constraints which are true for every operation.
        """
        return in_fmt in (DataFormat.Int8, DataFormat.UInt8) and in_fmt == out_fmt

    # Targeted dimensions per (dest_sync, dest_acc) that cover key corner cases:
    # 1 tile (minimum), max-wide (stresses block_ct), max-tall (stresses block_rt),
    # and max-square (both loops at capacity).
    unpack_reduce_col_tilizeA_strided_dims = {
        (DestSync.Half, DestAccumulation.No): [
            [32, 32],
            [32, 256],
            [256, 32],
            [64, 128],
        ],
        (DestSync.Half, DestAccumulation.Yes): [
            [32, 32],
            [32, 128],
            [128, 32],
            [64, 64],
        ],
        (DestSync.Full, DestAccumulation.No): [
            [32, 32],
            [32, 512],
            [512, 32],
            [128, 128],
        ],
        (DestSync.Full, DestAccumulation.Yes): [
            [32, 32],
            [32, 256],
            [256, 32],
            [64, 128],
        ],
    }

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
            for dest_sync in (DestSync.Half, DestSync.Full):
                for dimensions in unpack_reduce_col_tilizeA_strided_dims[
                    (dest_sync, acc)
                ]:
                    for pool_type in (
                        ReducePool.Max,
                        ReducePool.Sum,
                        ReducePool.Average,
                    ):
                        if pool_type == ReducePool.Average and in_fmt.is_integer():
                            continue
                        combinations.append(
                            (fmt, acc, dest_sync, runtime(dimensions), pool_type)
                        )

    return combinations


UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS = input_output_formats(
    [
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
        DataFormat.MxFp8P,
        DataFormat.MxFp8R,
        DataFormat.MxFp6P,
        DataFormat.MxFp6R,
        DataFormat.MxFp4,
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


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims=ALL_UNPACK_REDUCE_COL_TILIZEA_STRIDED_COMBINATIONS,
)
def test_unpack_reduce_col_tilizeA_strided_quasar(
    formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims,
    boot_mode=BootMode.DEFAULT,
):
    (formats, dest_acc, dest_sync_mode, input_dimensions, pool_type) = (
        formats_dest_acc_sync_unpack_reduce_col_tilizeA_strided_sel_dims[0]
    )

    num_faces = 4
    reduce_dim = ReduceDimension.Column
    math_fidelity = MathFidelity.LoFi

    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if pool_type == ReducePool.Average:
        src_B = torch.full((1024,), 1.0 / 32)
    else:
        src_B = torch.full((1024,), 1)

    tilize_gen = get_golden_generator(TilizeGolden)

    golden_src_A = src_A
    input_fmt = formats.input_format

    if formats.input_format.is_mx_format():
        golden_src_A = quantize_mx_tensor_chunked(src_A, formats.input_format)
        input_fmt = DataFormat.Float16_b

    golden_A = tilize_gen(
        golden_src_A, input_dimensions, formats.input_format, num_faces=num_faces
    )

    reduce_gen = get_golden_generator(ReduceGolden)
    golden_tensor = reduce_gen(
        golden_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        input_format=input_fmt,
    )

    mathop = {
        ReduceDimension.Row: MathOperation.ReduceRow,
        ReduceDimension.Column: MathOperation.ReduceColumn,
        ReduceDimension.Scalar: MathOperation.ReduceScalar,
    }[reduce_dim]

    configuration = TestConfig(
        "sources/quasar/unpack_reduce_col_tilizeA_strided_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(),
            NUM_FACES(),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=1,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
