# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.utils import passed_test

INPUT_DIMENSIONS = [[512, 32]]
# Nested list of [H, W] pairs: a flat [H, W] is expanded by parametrize into
# input_dimensions=H (int) and breaks generate_stimuli / rows, cols = dims.
PERF_INPUT_DIMENSIONS = [[512, 32]]
PERF_ONLY_INPUT_DIMENSIONS = [[512, 32]]
TILE_DIMENSIONS = [32, 32]


def unary_broadcast_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def unary_broadcast_implied_math_formats(formats, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    if formats.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def unary_broadcast_input_dimensions(*, is_perf=False):
    return PERF_ONLY_INPUT_DIMENSIONS if is_perf else INPUT_DIMENSIONS


UNARY_BROADCAST_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        # DataFormat.Float32, Buggy functionality for Float32 (unpack_to_dest=True) tbd
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,
)


def get_valid_dest_acc_unary_broadcast(formats):
    """Valid dest accumulation modes for unary broadcast."""
    accs = list(get_valid_dest_accumulation_modes(formats))
    if formats.input_format.is_32_bit():
        accs = [a for a in accs if a == DestAccumulation.Yes]
    elif formats.output_format == DataFormat.Float32:
        accs = [a for a in accs if a == DestAccumulation.Yes]
    return accs if accs else [DestAccumulation.Yes]


@pytest.mark.quasar
@parametrize(
    formats=UNARY_BROADCAST_FORMATS,
    dest_acc=lambda formats: get_valid_dest_acc_unary_broadcast(formats),
    broadcast_type=[
        BroadcastType.Scalar,
        BroadcastType.Column,
        BroadcastType.Row,
    ],
    implied_math_format=lambda formats: unary_broadcast_implied_math_formats(formats),
    dest_sync_mode=lambda: unary_broadcast_dest_sync_modes(is_perf=False),
    input_dimensions=runtime(lambda: unary_broadcast_input_dimensions(is_perf=False)),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unary_broadcast_quasar(
    formats,
    dest_acc,
    broadcast_type,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    rows, cols = input_dimensions
    num_elements = rows * cols
    tile_cnt = (rows // tile_rows) * (cols // tile_cols)

    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    torch_format = format_dict[formats.input_format]
    src_B = torch.randn(num_elements, dtype=torch_format)

    if not is_perf:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_B,
            formats.output_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt,
            face_r_dim=face_r_dim,
        )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    src_A = src_B

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_unary_broadcast_quasar_test.cpp",
        "formats": formats,
        "templates": [
            IMPLIED_MATH_FORMAT(implied_math_format),
            BROADCAST_TYPE(broadcast_type),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            NUM_TILES_IN_BLOCK(
                output_tiles_in_block,
                input_num_tiles_in_block=output_tiles_in_block,
                output_num_tiles_in_block=output_tiles_in_block,
            ),
            NUM_BLOCKS(
                output_num_blocks,
                input_num_blocks=output_num_blocks,
                output_num_blocks=output_num_blocks,
            ),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=FACE_C_DIM),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
        "boot_mode": boot_mode,
        "disable_format_inference": (implied_math_format == ImpliedMathFormat.Yes),
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    configuration = TestConfig(
        **{
            **test_config_kwargs,
            "templates": test_config_kwargs["templates"]
            + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=True
    )

    assert test_passed, "Assert against golden failed"
