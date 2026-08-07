# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test pack operation with various configurations.

Tests the LLK pack kernel with:
- Different data formats (Float16_b, Float16, Float32, Int32, Bfp8_b)
- Destination accumulation modes
- Variable tile dimensions
- ReLU activation
- Destination sync modes (SyncHalf for double-buffering, SyncFull for single-buffering)
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_dest_indices,
)
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    FACES_PER_TILE,
    TILE_DIMENSIONS,
    PackGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    PackerReluType,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    input_dimensions=[[32, 32], [64, 64], [32, 64], [64, 32]],
    relu_type=[
        PackerReluType.NoRelu,
        PackerReluType.ZeroRelu,
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ],
    dest_sync=[DestSync.Half, DestSync.Full],
    dest_index=lambda dest_acc, dest_sync, formats, input_dimensions: get_valid_dest_indices(
        dest_sync, dest_acc, formats, input_dimensions
    ),
)
def test_pack(
    formats,
    dest_acc,
    input_dimensions,
    relu_type,
    dest_sync,
    dest_index,
):
    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip(
            "Pack does not support mixing Int32 with other formats. Check format conversions in packer for more information."
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Generate golden output
    generate_golden = get_golden_generator(PackGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions=input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    # To come as close as possible to actual hardware behavior, we infer data formats here
    # and use the inferred pack_src format for ReLU operations.
    data_formats = infer_data_formats(
        input_format=formats.input_format,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
        unpacking_to_dest=unpack_to_dest,
    )

    # This is a bug in infer_pack_in function for blackhole. Force Float32 intermediate for DestAccumulation.Yes
    # TODO: fix infer_pack_in for blackhole.
    if (
        dest_acc == DestAccumulation.Yes
        and get_chip_architecture() == ChipArchitecture.BLACKHOLE
        and not formats.input_format.is_integer()
    ):
        data_formats.pack_src = DataFormat.Float32

    if data_formats.pack_src.is_integer() and relu_type in [
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]:
        pytest.skip(
            "Pack source format cannot be an integer format with ReLu Type: "
            + str(relu_type)
        )

    tensor_average = (
        torch.mean(golden_tensor).item()
        if not formats.output_format.is_integer()
        else 0.0
    )

    relu_config = PackGolden.generate_relu_config(
        relu_type,
        relu_threshold=tensor_average,  # We use the average value for this.
        intermediate_format=data_formats.pack_src,
    )

    # Perform relu.
    golden_tensor = PackGolden.apply_relu(
        golden_tensor,
        relu_config,
        data_formats.pack_src,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    if dest_index != 0:
        num_tiles_in_block = num_tiles_in_block - dest_index
        if num_tiles_in_block <= 0 or tile_cnt_A % num_tiles_in_block != 0:
            pytest.skip(
                f"Dest index {dest_index} is not valid for tile count {tile_cnt_A} and num_tiles_in_block {num_tiles_in_block}."
            )
        num_blocks = tile_cnt_A // num_tiles_in_block

    configuration = TestConfig(
        "sources/pack_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX(dest_index),
            RELU_CONFIG(relu_config),
            NUM_FACES(num_faces=FACES_PER_TILE),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=False
    )

    if (
        not test_passed
        and relu_type
        in [
            PackerReluType.MinThresholdRelu,
            PackerReluType.MaxThresholdRelu,
        ]
        and PackGolden.is_relu_threshold_tolerance_issue(
            golden_tensor,
            res_tensor,
            relu_config,
            data_formats.pack_src,
        )
    ):
        # When a datum is extremely close to the ReLU threshold, differences can arise due to
        # floating point precision limitations and rounding during format conversions.
        # We check if all mismatches are within a small tolerance of the threshold. If so, we consider the test as passed.
        test_passed = True

    assert test_passed
