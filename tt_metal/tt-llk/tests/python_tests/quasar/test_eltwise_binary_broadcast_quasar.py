# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    generate_reduced_input_dimensions,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    quasar_mx_smoke,
    runtime,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BootMode
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import SUPPORTED_TILE_SIZES, is_mx_unsupported_tile_dims
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test
from quasar.test_eltwise_binary_quasar import (
    get_num_tiles_per_accumulation,
    valid_acc_to_dest,
)

_BINARY_BROADCAST_BASE_FORMATS = [
    DataFormat.Float16_b,
    DataFormat.Float16,
]
_BINARY_BROADCAST_EXPLICIT_FORMATS = [
    InputOutputFormat(DataFormat.Int8, DataFormat.Int32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
]
BINARY_BROADCAST_FORMATS = (
    input_output_formats(_BINARY_BROADCAST_BASE_FORMATS)
    + _BINARY_BROADCAST_EXPLICIT_FORMATS
    + quasar_mx_smoke(DataFormat.MxFp4, DataFormat.Float16_b)
)
BINARY_BROADCAST_PERF_FORMATS = (
    input_output_formats(
        _BINARY_BROADCAST_BASE_FORMATS,
        same=True,
    )
    + _BINARY_BROADCAST_EXPLICIT_FORMATS
    + quasar_mx_smoke(DataFormat.MxFp4, DataFormat.Float16_b)
)

BROADCAST_TYPES = [
    BroadcastType.Column,
    BroadcastType.Row,
    BroadcastType.Scalar,
]


def binary_broadcast_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def binary_broadcast_tile_dimensions(formats, broadcast_type, *, is_perf=False):
    tile_sizes = (
        select_perf_tile_sizes(SUPPORTED_TILE_SIZES)
        if is_perf
        else SUPPORTED_TILE_SIZES
    )
    return [
        list(tile_dims)
        for tile_dims in tile_sizes
        if not (
            broadcast_type in (BroadcastType.Row, BroadcastType.Column)
            and tuple(tile_dims) == (32, 16)
        )
        and not is_mx_unsupported_tile_dims(
            formats.input_format, formats.output_format, tile_dims
        )
    ]


def binary_broadcast_input_dimensions(
    dest_acc, dest_sync, tile_dimensions, *, is_perf=False
):
    tile_shape = construct_tile_shape(tile_dimensions)
    if is_perf:
        return generate_perf_input_dimensions(dest_acc, dest_sync, tile_shape)
    return generate_reduced_input_dimensions(dest_acc, dest_sync, tile_shape)


def binary_broadcast_acc_to_dest_modes(
    input_dimensions, tile_dimensions, *, is_perf=False
):
    if is_perf and tuple(tile_dimensions) == (1, 32) and input_dimensions[0] == 1:
        return [False]
    return valid_acc_to_dest(input_dimensions, tile_dimensions)


def binary_broadcast_implied_math_formats(format, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    if format.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def binary_broadcast_math_fidelities(format, math_op):
    # Int8 is an exact integer op. Float16_b is full precision at LoFi.
    if format.input_format in (DataFormat.Int8, DataFormat.Float16_b):
        return [MathFidelity.LoFi]
    return get_valid_math_fidelities(format, math_op)


@pytest.mark.quasar
@parametrize(
    formats=BINARY_BROADCAST_FORMATS,
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    math_op=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    broadcast_type=BROADCAST_TYPES,
    math_fidelity=lambda formats, math_op: binary_broadcast_math_fidelities(
        formats, math_op
    ),
    implied_math_format=lambda formats: binary_broadcast_implied_math_formats(formats),
    dest_sync=lambda: binary_broadcast_dest_sync_modes(is_perf=False),
    unpack_to_dest=[False],
    tile_dimensions=lambda formats, broadcast_type: binary_broadcast_tile_dimensions(
        formats, broadcast_type, is_perf=False
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync, tile_dimensions: binary_broadcast_input_dimensions(
            dest_acc, dest_sync, tile_dimensions, is_perf=False
        )
    ),
    acc_to_dest=lambda input_dimensions, tile_dimensions: binary_broadcast_acc_to_dest_modes(
        input_dimensions, tile_dimensions, is_perf=False
    ),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_eltwise_binary_broadcast_quasar(
    formats,
    dest_acc,
    math_op,
    broadcast_type,
    math_fidelity,
    implied_math_format,
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

    num_blocks, input_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
    )
    output_tiles_in_block = input_tiles_in_block // num_tiles_per_accumulation

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
    tile_cnt_res = tile_cnt_A // num_tiles_per_accumulation

    generate_broadcast_golden = get_golden_generator(BroadcastGolden)
    bcast_src_B_tensor = generate_broadcast_golden(
        broadcast_type,
        src_B,
        formats.output_format,
        num_faces=num_faces,
        tile_cnt=tile_cnt_A,
        face_r_dim=tile_shape.face_r_dim,
        input_format=formats.input_format,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    input_format = formats.input_format
    input_format_B = (
        DataFormat.Float16_b
        if formats.input_format.is_mx_format()
        else formats.input_format
    )
    golden_tensor = generate_golden(
        math_op,
        src_A,
        bcast_src_B_tensor,
        formats.output_format,
        math_fidelity,
        input_format=input_format,
        input_format_B=input_format_B,
        acc_to_dest=acc_to_dest,
        tile_shape=tile_shape,
        num_tiles_per_accumulation=num_tiles_per_accumulation,
        dest_acc=dest_acc,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_binary_broadcast_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=math_op),
            IMPLIED_MATH_FORMAT(implied_math_format),
            BROADCAST_TYPE(broadcast_type),
            DEST_SYNC(dest_sync),
            ACC_TO_DEST(acc_to_dest),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                tile_dimensions=tile_dimensions,
            ),
            TILE_COUNT(tile_cnt_A),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_res),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(
                input_tiles_in_block,
                output_num_tiles_in_block=output_tiles_in_block,
            ),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
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

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
