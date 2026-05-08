# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import PackGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    ReluConfig,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    RELU_CONFIG,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.utils import passed_test

INPUT_DIMENSIONS = [[512, 64], [192, 512]]
TILE_DIMENSIONS = [32, 32]
# Complete list of formats that are supported with L1 accumulation
PACK_L1_ACC_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Int8,
        DataFormat.UInt8,
    ]
)


def generate_qsr_pack_l1_acc_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate pack combinations for Quasar pack with L1 accumulation tests.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    def is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
        """Check if the format conversion is supported by packer. These format conversions are dependent on the dest register mode."""
        # Upcasting to Float32/Int32 requires dest_acc enabled
        if (
            out_fmt.is_32_bit()
            and not in_fmt.is_32_bit()
            and dest_acc == DestAccumulation.No
        ):
            return False
        # Int8<->UInt8 conversion requires dest_acc enabled
        if (
            dest_acc == DestAccumulation.No
            and in_fmt in (DataFormat.Int8, DataFormat.UInt8)
            and in_fmt != out_fmt
        ):
            return False
        return True

    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        for dest_acc in get_dest_acc_modes(in_fmt):
            if is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
                combinations.append((fmt, dest_acc))

    return combinations


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=generate_qsr_pack_l1_acc_combinations(PACK_L1_ACC_FORMATS),
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    input_dimensions=INPUT_DIMENSIONS,
)
def test_pack_l1_acc_quasar(
    formats_dest_acc,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    boot_mode=BootMode.DEFAULT,
):
    (formats, dest_acc) = formats_dest_acc

    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    rows, cols = input_dimensions
    tile_cnt = (rows // tile_rows) * (cols // tile_cols)

    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    src_A, _, src_B, _ = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMENSIONS,
    )

    generate_golden = get_golden_generator(PackGolden)
    full_golden = generate_golden(
        src_A,
        formats.output_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
        face_r_dim=face_r_dim,
    )

    # This test accumulates the results of each block on top of each other
    # Slice the full golden into per-block partials and accumulate
    elements_per_block = output_tiles_in_block * num_faces * face_r_dim * FACE_C_DIM
    partials = [
        full_golden[block * elements_per_block : (block + 1) * elements_per_block]
        for block in range(output_num_blocks)
    ]
    golden_tensor = generate_golden.accumulate_l1(
        partials, data_format=formats.output_format
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/pack_l1_acc_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(implied_math_format),
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
            RELU_CONFIG(ReluConfig.NoRelu.value),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=output_tiles_in_block,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
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
