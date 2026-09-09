# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BootMode
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    TEST_FACE_DIMS,
    generate_input_dim,
)
from helpers.tile_constants import SUPPORTED_TILE_SIZES, is_mx_unsupported_tile_dims
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test


def eltwise_binary_dest_acc(formats):
    return (
        [DestAccumulation.Yes]
        if formats.input_format == DataFormat.Int8
        else [DestAccumulation.No]
    )


def eltwise_binary_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def eltwise_binary_implied_math_formats(formats, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    if formats.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def eltwise_binary_math_fidelities(math_op, formats):
    if (
        math_op in [MathOperation.Elwadd, MathOperation.Elwsub]
        or formats.input_format == DataFormat.Int8
    ):
        return [MathFidelity.LoFi]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


# For acc_to_dest setting, accumulate two result tiles into dest. Can be extended.
def get_num_tiles_per_accumulation(acc_to_dest: bool) -> int:
    return 2 if acc_to_dest else 1


def eltwise_binary_tile_dimensions(formats, *, is_perf=False):
    tile_sizes = (
        select_perf_tile_sizes(SUPPORTED_TILE_SIZES)
        if is_perf
        else SUPPORTED_TILE_SIZES
    )
    return [
        list(tile_dims)
        for tile_dims in tile_sizes
        if not is_mx_unsupported_tile_dims(
            formats.input_format, formats.output_format, tile_dims
        )
    ]


def eltwise_binary_input_dimensions(
    dest_acc, dest_sync, tile_dimensions, *, is_perf=False
):
    tile_shape = construct_tile_shape(tile_dimensions)
    if is_perf:
        return generate_perf_input_dimensions(dest_acc, dest_sync, tile_shape)
    return generate_unary_input_dimensions(dest_acc, dest_sync, tile_shape)


def valid_acc_to_dest(input_dimensions, tile_dimensions) -> list:
    """Pick the acc_to_dest modes worth running for a given input size.

    acc_to_dest=True accumulates `get_num_tiles_per_accumulation(True)` result tiles into
    dest, so it only makes sense when the tile count is a non-zero multiple of that.
    """
    tile_rows, tile_cols = tile_dimensions
    tile_count = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    per_acc = get_num_tiles_per_accumulation(True)
    if tile_count >= per_acc and tile_count % per_acc == 0:
        return [False, True]
    return [False]


ELTWISE_FORMATS = input_output_formats(
    [
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
        DataFormat.Float16_b,
        DataFormat.Float16,
    ],
) + [InputOutputFormat(DataFormat.Int8, DataFormat.Int32)]


@pytest.mark.quasar
@parametrize(
    formats=ELTWISE_FORMATS,
    math_op=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    math_fidelity=eltwise_binary_math_fidelities,
    implied_math_format=lambda formats: eltwise_binary_implied_math_formats(
        formats, is_perf=False
    ),
    dest_acc=eltwise_binary_dest_acc,
    dest_sync=lambda: eltwise_binary_dest_sync_modes(is_perf=False),
    unpack_to_dest=[False],
    tile_dimensions=lambda formats: eltwise_binary_tile_dimensions(
        formats, is_perf=False
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync, tile_dimensions: eltwise_binary_input_dimensions(
            dest_acc, dest_sync, tile_dimensions, is_perf=False
        )
    ),
    acc_to_dest=valid_acc_to_dest,
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_eltwise_binary(
    formats,
    math_op,
    math_fidelity,
    implied_math_format,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    tile_dimensions,
    input_dimensions,
    acc_to_dest,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    tile_shape = construct_tile_shape(tile_dimensions)
    num_faces = tile_shape.total_num_faces()
    num_tiles_per_accumulation = get_num_tiles_per_accumulation(acc_to_dest)

    if formats.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=formats.output_format,
        tile_dimensions=tile_dimensions,
    )

    tile_cnt_res = src_A.numel() // (
        tile_shape.total_tile_size() * num_tiles_per_accumulation
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        math_op,
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_format=formats.input_format,
        acc_to_dest=acc_to_dest,
        tile_shape=tile_shape,
        num_tiles_per_accumulation=num_tiles_per_accumulation,
        dest_acc=dest_acc,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_binary_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=math_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync),
            ACC_TO_DEST(acc_to_dest),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=tile_dimensions
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_res),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_TILES_IN_BLOCK(num_tiles_per_accumulation),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_res,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
        "disable_format_inference": formats.input_format.is_mx_format(),
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

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"
