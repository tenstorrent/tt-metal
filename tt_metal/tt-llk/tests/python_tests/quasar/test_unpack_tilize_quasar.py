# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TilizeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.perf import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tile_constants import (
    MX_SUPPORTED_TILE_SIZES,
    is_mx_unsupported_tile_dims,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

UNPACK_TILIZE_TILE_SIZES = [
    (32, 32),
    (1, 32),
    (2, 32),
]


def unpack_tilize_formats(formats_list, *, is_perf=False):
    return formats_list


def unpack_tilize_dest_acc_modes(formats):
    if formats.input_format.is_32_bit():
        return [DestAccumulation.Yes]
    if formats.input_format in (DataFormat.Float16, DataFormat.Int16):
        return [DestAccumulation.No]
    return [DestAccumulation.No, DestAccumulation.Yes]


def unpack_tilize_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def unpack_tilize_unpacker_engines(formats, dest_acc):
    candidates = (
        [UnpackerEngine.UnpDest]
        if formats.input_format.is_32_bit()
        else [UnpackerEngine.UnpA, UnpackerEngine.UnpB]
    )
    return [
        unpacker_sel
        for unpacker_sel in candidates
        if not (
            dest_acc == DestAccumulation.Yes and unpacker_sel == UnpackerEngine.UnpB
        )
    ]


def unpack_tilize_tile_dimensions(formats, dest_acc, *, is_perf=False):
    return [
        tile_dims
        for tile_dims in UNPACK_TILIZE_TILE_SIZES
        if not is_mx_unsupported_tile_dims(
            formats.input_format, formats.output_format, tile_dims
        )
        and not (
            formats.input_format.is_32_bit()
            and dest_acc == DestAccumulation.Yes
            and tile_dims not in MX_SUPPORTED_TILE_SIZES
        )
    ]


def unpack_tilize_input_dimensions(
    dest_acc, dest_sync_mode, tile_dimensions, *, is_perf=False
):
    if is_perf:
        return generate_perf_input_dimensions(dest_acc)
    return generate_unary_input_dimensions(
        dest_acc,
        dest_sync=dest_sync_mode,
        tile_shape=construct_tile_shape(tile_dimensions),
    )


UNPACK_TILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,  # Input format and output format are the same
)


@pytest.mark.quasar
@parametrize(
    formats=lambda: unpack_tilize_formats(UNPACK_TILIZE_FORMATS, is_perf=False),
    dest_acc=unpack_tilize_dest_acc_modes,
    dest_sync_mode=lambda: unpack_tilize_dest_sync_modes(is_perf=False),
    unpacker_sel=unpack_tilize_unpacker_engines,
    tile_dimensions=lambda formats, dest_acc: unpack_tilize_tile_dimensions(
        formats, dest_acc, is_perf=False
    ),
    input_dimensions=lambda dest_acc, dest_sync_mode, tile_dimensions: unpack_tilize_input_dimensions(
        dest_acc, dest_sync_mode, tile_dimensions, is_perf=False
    ),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unpack_tilize_quasar(
    formats,
    dest_acc,
    dest_sync_mode,
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

    sequential_spec = StimuliSpec.sequential()
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=sequential_spec,
        spec_B=sequential_spec,
    )

    num_faces = tile_shape.total_num_faces()

    generate_golden = get_golden_generator(TilizeGolden)
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    if formats.input_format.is_mx_format():
        golden_src = quantize_mx_tensor_chunked(golden_src, formats.input_format)
    golden_tensor = generate_golden(
        golden_src,
        input_dimensions,
        formats.output_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/unpack_tilize_quasar_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=tile_dimensions
            ),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
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
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
