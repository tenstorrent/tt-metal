# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    Transpose,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (  # generate_stimuli_w_tile_dimensions
    generate_stimuli,
)
from helpers.test_config import BootMode
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tile_constants import (
    MX_SUPPORTED_TILE_SIZES,
    SUPPORTED_TILE_SIZES,
    is_mx_unsupported_tile_dims,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test


def unpack_unary_formats(formats_list, *, is_perf=False):
    return formats_list


def unpack_unary_dest_acc_modes(formats, *, is_perf=False):
    candidates = (
        [DestAccumulation.Yes]
        if formats.input_format.is_32_bit()
        else [DestAccumulation.No, DestAccumulation.Yes]
    )
    return [
        dest_acc
        for dest_acc in candidates
        if not (
            formats.input_format != DataFormat.Float32
            and formats.output_format == DataFormat.Float32
            and dest_acc == DestAccumulation.No
        )
    ]


def unpack_unary_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def unpack_unary_transpose_modes(formats):
    if formats.input_format.is_32_bit():
        return [Transpose.No]
    return [Transpose.No, Transpose.Yes]


def unpack_unary_engines(formats):
    if formats.input_format.is_32_bit():
        return [UnpackerEngine.UnpDest]
    return [UnpackerEngine.UnpA, UnpackerEngine.UnpB]


def unpack_unary_tile_dimensions(formats, transpose, unpacker_sel, *, is_perf=False):
    if transpose == Transpose.Yes:
        candidates = [(32, 32)]
    else:
        candidates = SUPPORTED_TILE_SIZES
    return [
        tile_dims
        for tile_dims in candidates
        if not is_mx_unsupported_tile_dims(
            formats.input_format, formats.output_format, tile_dims
        )
        and not (
            unpacker_sel == UnpackerEngine.UnpDest
            and tile_dims not in MX_SUPPORTED_TILE_SIZES
        )
    ]


def unpack_unary_input_dimensions(
    dest_acc, dest_sync_mode, tile_dimensions, *, is_perf=False
):
    if is_perf:
        return generate_perf_input_dimensions(dest_acc)
    return generate_unary_input_dimensions(
        dest_acc,
        dest_sync=dest_sync_mode,
        tile_shape=construct_tile_shape(tile_dimensions),
    )


UNPACK_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats=lambda: unpack_unary_formats(UNPACK_FORMATS, is_perf=False),
    dest_acc=lambda formats: unpack_unary_dest_acc_modes(formats, is_perf=False),
    dest_sync_mode=lambda: unpack_unary_dest_sync_modes(is_perf=False),
    transpose=unpack_unary_transpose_modes,
    unpacker_sel=unpack_unary_engines,
    tile_dimensions=runtime(
        lambda formats, transpose, unpacker_sel: unpack_unary_tile_dimensions(
            formats, transpose, unpacker_sel, is_perf=False
        )
    ),
    input_dimensions=runtime(
        lambda dest_acc, dest_sync_mode, tile_dimensions: unpack_unary_input_dimensions(
            dest_acc, dest_sync_mode, tile_dimensions, is_perf=False
        )
    ),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unpack_unary_operand_quasar(
    formats,
    dest_acc,
    dest_sync_mode,
    transpose,
    unpacker_sel,
    input_dimensions,
    tile_dimensions,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    tile_shape = construct_tile_shape(tile_dimensions)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    num_faces = tile_shape.total_num_faces()

    golden_src = (
        src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    )  # use A for UnpA and UnpDest
    if transpose == Transpose.Yes:
        if formats.input_format.is_mx_format():
            golden_src = quantize_mx_tensor_chunked(golden_src, formats.input_format)

        generate_golden = get_golden_generator(TransposeGolden)
        golden_tensor = generate_golden.transpose_faces_multi_tile(
            golden_src,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden_tensor = generate_golden.transpose_within_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=input_dimensions,
        )
        # TransposeGolden only rearranges; it doesn't round-trip through the
        # output MX lattice the way DataCopyGolden does. For MxFp4 -> MxInt4
        # the MxFp4 lattice has 1.5 but MxInt4 (with the realized block scale)
        # may not, so HW rounds 1.5 -> 2.0 while golden keeps 1.5. Snap golden
        # to the output lattice here to match.
        if formats.output_format.is_mx_format():
            golden_tensor = quantize_mx_tensor_chunked(
                golden_tensor.to(torch.bfloat16), formats.output_format
            )
    else:
        generate_golden = get_golden_generator(DataCopyGolden)
        golden_tensor = generate_golden(
            golden_src,
            formats.output_format,
            num_faces=num_faces,
            input_dimensions=input_dimensions,
            input_format=formats.input_format,
            face_r_dim=tile_shape.face_r_dim,
            tile_shape=tile_shape,
        )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/unpack_unary_operand_quasar_test.cpp",
        "formats": formats,
        "templates": [
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync_mode),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                tile_dimensions=tile_dimensions,
            ),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
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
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": (
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
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
        golden_tensor, res_tensor, formats.output_format, tile_shape=tile_shape
    ), "Assert against golden failed"
