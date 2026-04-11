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
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.utils import passed_test

INPUT_DIMENSIONS = [[512, 32]]
TILE_DIMENSIONS = [32, 32]


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
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            # DataFormat.Float32, Buggy functionality for Float32 (unpack_to_dest=True) tbd
            DataFormat.MxFp8R,
            DataFormat.MxFp8P,
        ],
        same=True,
    ),
    dest_acc=lambda formats: get_valid_dest_acc_unary_broadcast(formats),
    broadcast_type=[
        BroadcastType.Scalar,
        BroadcastType.Column,
        BroadcastType.Row,
    ],
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    input_dimensions=INPUT_DIMENSIONS,
)
def test_unary_broadcast_quasar(
    formats,
    dest_acc,
    broadcast_type,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    boot_mode=BootMode.DEFAULT,
):
    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    rows, cols = input_dimensions
    num_elements = rows * cols
    tile_cnt = (rows // tile_rows) * (cols // tile_cols)

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else DestAccumulation.No
    )
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        effective_dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    torch_format = format_dict[formats.input_format]
    src_B = torch.randn(num_elements, dtype=torch_format)

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

    configuration = TestConfig(
        "sources/quasar/eltwise_unary_broadcast_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(implied_math_format),
            BROADCAST_TYPE(broadcast_type),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
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
        ],
        variant_stimuli=StimuliConfig(
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
        disable_format_inference=(implied_math_format == ImpliedMathFormat.Yes),
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
