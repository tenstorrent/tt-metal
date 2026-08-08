# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import (
    get_valid_data_format_conversions,
    get_valid_dest_accumulation_modes,
)
from helpers.format_config import DataFormat
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
    generate_perf_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf import create_test_or_perf_config
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
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

_UNPACK_REDUCE_COL_TILIZEA_STRIDED_DIMS = {
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


def unpack_reduce_col_tilizeA_strided_formats(formats_list):
    return [
        formats
        for formats in get_valid_data_format_conversions(formats_list)
        if formats.input_format != DataFormat.Int32
    ]


def unpack_reduce_col_tilizeA_strided_dest_acc_modes(formats):
    requires_dest_acc = (
        formats.input_format in (DataFormat.Int8, DataFormat.UInt8)
        and formats.input_format == formats.output_format
    )
    return [
        dest_acc
        for dest_acc in get_valid_dest_accumulation_modes(formats)
        if not (requires_dest_acc and dest_acc == DestAccumulation.No)
    ]


def unpack_reduce_col_tilizeA_strided_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def unpack_reduce_col_tilizeA_strided_dimensions(
    dest_acc, dest_sync_mode, *, is_perf=False
):
    if is_perf:
        return generate_perf_input_dimensions(dest_acc)
    return _UNPACK_REDUCE_COL_TILIZEA_STRIDED_DIMS[(dest_sync_mode, dest_acc)]


def unpack_reduce_col_tilizeA_strided_pool_types(formats):
    pools = [ReducePool.Max, ReducePool.Sum, ReducePool.Average]
    if formats.input_format.is_integer():
        return [pool for pool in pools if pool != ReducePool.Average]
    return pools


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


def unpack_reduce_col_tilizeA_strided_implied_math_formats():
    return [ImpliedMathFormat.No]


@pytest.mark.quasar
@parametrize(
    formats=lambda: unpack_reduce_col_tilizeA_strided_formats(
        UNPACK_REDUCE_COL_TILIZEA_STRIDED_FORMATS
    ),
    dest_acc=unpack_reduce_col_tilizeA_strided_dest_acc_modes,
    dest_sync_mode=lambda: unpack_reduce_col_tilizeA_strided_dest_sync_modes(
        is_perf=False
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode: unpack_reduce_col_tilizeA_strided_dimensions(
            dest_acc, dest_sync_mode, is_perf=False
        )
    ),
    pool_type=unpack_reduce_col_tilizeA_strided_pool_types,
    implied_math_format=unpack_reduce_col_tilizeA_strided_implied_math_formats,
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unpack_reduce_col_tilizeA_strided_quasar(
    formats,
    dest_acc,
    dest_sync_mode,
    input_dimensions,
    pool_type,
    implied_math_format,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
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

    if not is_perf:
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
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(),
            NUM_FACES(),
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
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
