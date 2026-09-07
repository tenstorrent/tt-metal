# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.dest_params import (
    UnpackPath,
    dest_acc_modes,
    dest_sync_modes,
    unpack_to_dest_modes,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
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

PACK_UNTILIZE_TILE_SIZES = [
    (32, 32),
    (1, 32),
    (2, 32),
]


def _pack_untilize_supported(formats):
    in_fmt, out_fmt = formats.input_format, formats.output_format
    if in_fmt.is_integer() ^ out_fmt.is_integer():
        return False
    if (in_fmt == DataFormat.Int16) ^ (out_fmt == DataFormat.Int16):
        return False
    if out_fmt.is_mx_format():
        return False
    return True


PACK_UNTILIZE_FORMATS = [
    fmt
    for fmt in input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Int16,
            DataFormat.Int32,
            DataFormat.MxFp4,
            DataFormat.MxInt8,
            DataFormat.MxInt4,
            DataFormat.MxInt2,
        ]
    )
    if _pack_untilize_supported(fmt)
]


def pack_untilize_tile_dimensions(formats, dest_acc, *, is_perf=False):
    in_fmt, out_fmt = formats.input_format, formats.output_format
    tile_sizes = (
        select_perf_tile_sizes(PACK_UNTILIZE_TILE_SIZES)
        if is_perf
        else PACK_UNTILIZE_TILE_SIZES
    )
    selected = []
    for tile_dims in tile_sizes:
        if is_mx_unsupported_tile_dims(in_fmt, out_fmt, tile_dims):
            continue
        if (
            in_fmt.is_32_bit()
            and dest_acc == DestAccumulation.Yes
            and tile_dims not in MX_SUPPORTED_TILE_SIZES
        ):
            continue
        selected.append(list(tile_dims))
    return selected


def pack_untilize_input_dimensions(
    dest_acc, dest_sync, tile_dimensions, *, is_perf=False
):
    tile_shape = construct_tile_shape(tile_dimensions)
    if is_perf:
        return generate_perf_input_dimensions(
            dest_acc, dest_sync, tile_shape, use_largest_fallback=True
        )
    return generate_unary_input_dimensions(dest_acc, dest_sync, tile_shape)


def pack_untilize_unpack_to_dest(formats, dest_acc):
    return unpack_to_dest_modes(
        formats,
        dest_acc,
        path=(
            UnpackPath.ForceTrue
            if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
            else UnpackPath.ForceFalse
        ),
    )


@pytest.mark.quasar
@parametrize(
    formats=PACK_UNTILIZE_FORMATS,
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=False),
    unpack_to_dest=pack_untilize_unpack_to_dest,
    tile_dimensions=lambda formats, dest_acc: pack_untilize_tile_dimensions(
        formats, dest_acc, is_perf=False
    ),
    input_dimensions=lambda dest_acc, dest_sync, tile_dimensions: pack_untilize_input_dimensions(
        dest_acc, dest_sync, tile_dimensions, is_perf=False
    ),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_pack_untilize_quasar(
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    tile_dimensions,
    input_dimensions,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    dest_sync_mode = dest_sync

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

    generate_golden = get_golden_generator(UntilizeGolden)
    if not is_perf:
        golden_tensor = generate_golden(
            src_A,
            formats.output_format,
            input_dimensions,
            input_format=formats.input_format,
            tile_dimensions=tile_dimensions,
        )

    num_faces = tile_shape.total_num_faces()

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/pack_untilize_quasar_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=tile_dimensions
            ),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            UNPACKER_ENGINE_SEL(),
        ],
        "runtimes": [
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
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
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
        "disable_format_inference": (
            formats.input_format.is_mx_format() or formats.output_format.is_mx_format()
        ),
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
