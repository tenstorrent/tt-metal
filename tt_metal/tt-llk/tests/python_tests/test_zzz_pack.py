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


def _valid_relu_types(formats, dest_acc):
    """Return relu types compatible with the inferred pack_src format.

    Threshold-based relu (MinThresholdRelu, MaxThresholdRelu) is not supported
    when pack_src is an integer format, so those types are excluded for such
    combinations.
    """
    all_relu_types = [
        PackerReluType.NoRelu,
        PackerReluType.ZeroRelu,
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    try:
        data_formats = infer_data_formats(
            input_format=formats.input_format,
            output_format=formats.output_format,
            is_fp32_dest_acc_en=dest_acc,
            unpacking_to_dest=unpack_to_dest,
        )
    except ValueError:
        return []

    # Blackhole workaround: force Float32 intermediate for DestAccumulation.Yes
    if (
        dest_acc == DestAccumulation.Yes
        and get_chip_architecture() == ChipArchitecture.BLACKHOLE
        and not formats.input_format.is_integer()
    ):
        data_formats.pack_src = DataFormat.Float32

    if data_formats.pack_src.is_integer():
        return [
            rt
            for rt in all_relu_types
            if rt
            not in [PackerReluType.MinThresholdRelu, PackerReluType.MaxThresholdRelu]
        ]

    return all_relu_types


def _valid_dest_indices(dest_acc, dest_sync, formats, input_dimensions):
    """Return only dest indices that produce a valid block decomposition.

    For non-zero dest_index values, num_tiles_in_block is reduced by
    dest_index.  If the reduced value is non-positive or does not evenly
    divide tile_cnt_A, the index is invalid and excluded.
    """
    indices = get_valid_dest_indices(dest_sync, dest_acc, formats, input_dimensions)

    tile_cnt_A = (input_dimensions[0] // TILE_DIMENSIONS[0]) * (
        input_dimensions[1] // TILE_DIMENSIONS[1]
    )

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    valid = []
    for idx in indices:
        if idx == 0:
            valid.append(idx)
        else:
            adjusted = num_tiles_in_block - idx
            if adjusted > 0 and tile_cnt_A % adjusted == 0:
                valid.append(idx)

    return valid


def is_relu_threshold_tolerance_issue(
    golden_tensor,
    result_tensor,
    relu_config,
    intermediate_format,
    rtol=0.01,
    atol=0.01,
):
    """
    Check if test failure is due to threshold rounding/format conversion issues in ReLU.
    When a value is very close to the threshold, golden (Python) and hardware (Tensix)
    may make different decisions due to:
    - FP16/BF16 precision differences
    - Rounding during format conversions
    - Threshold encoding/decoding precision loss
    With values relatively close to the threshold, these small differences can lead to
    one side being clamped to zero while the other retains a small non-zero value.
    This function checks if all mismatches between golden and result tensors
    can be explained by such near-threshold issues.
    Args:
        golden_tensor: Expected output tensor
        result_tensor: Actual hardware output tensor
        relu_config: The ReLU configuration value
        rtol: Relative tolerance for threshold proximity checks (default 0.01)
        atol: Absolute tolerance for threshold proximity checks (default 0.01)
    Returns:
        bool: True if all mismatches are near-threshold rounding issues, False otherwise
    """
    relu_type = PackGolden.get_relu_type(relu_config)
    threshold = PackGolden.get_relu_threshold(relu_config, intermediate_format)

    # Only applicable for threshold-based ReLU modes
    # Zero relu is exact because of the sign bit, so no tolerance issues there.
    if relu_type not in [
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]:
        return False

    mismatches = ~torch.isclose(golden_tensor, result_tensor, rtol=rtol, atol=atol)

    if not mismatches.any():
        return False

    # Check if values are within tolerance of the threshold
    golden_near_threshold = torch.isclose(
        golden_tensor[mismatches],
        torch.full_like(golden_tensor[mismatches], threshold),
        rtol=rtol,
        atol=atol,
    )
    result_near_threshold = torch.isclose(
        result_tensor[mismatches],
        torch.full_like(result_tensor[mismatches], threshold),
        rtol=rtol,
        atol=atol,
    )

    acceptable = False
    if relu_type == PackerReluType.MinThresholdRelu:
        # One side should be 0, other should be near threshold
        golden_is_zero = golden_tensor[mismatches] == 0.0
        result_is_zero = result_tensor[mismatches] == 0.0
        acceptable = (golden_is_zero & result_near_threshold) | (
            result_is_zero & golden_near_threshold
        )
    else:  # For MAX_THRESHOLD_RELU: Check if both values are near the threshold
        acceptable = golden_near_threshold & result_near_threshold

    return acceptable.all().item()


@parametrize(
    formats=[
        f
        for f in input_output_formats(
            [
                DataFormat.Float16_b,
                DataFormat.Float16,
                DataFormat.Float32,
                DataFormat.Int32,
                DataFormat.Bfp8_b,
            ]
        )
        if not (
            (f.input_format == DataFormat.Int32) ^ (f.output_format == DataFormat.Int32)
        )
    ],
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    input_dimensions=[[32, 32], [64, 64], [32, 64], [64, 32]],
    dest_sync=[DestSync.Half, DestSync.Full],
    relu_type=lambda formats, dest_acc: _valid_relu_types(formats, dest_acc),
    dest_index=lambda dest_acc, dest_sync, formats, input_dimensions: _valid_dest_indices(
        dest_acc, dest_sync, formats, input_dimensions
    ),
)
def test_pack(
    formats,
    dest_acc,
    input_dimensions,
    dest_sync,
    relu_type,
    dest_index,
):

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
        and is_relu_threshold_tolerance_issue(
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
