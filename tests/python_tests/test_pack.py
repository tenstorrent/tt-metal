# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
from helpers.golden_generators import PackGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, PackerReluType, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    INPUT_DIMENSIONS,
    NUM_FACES,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
)
from helpers.utils import passed_test


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
    dest_index=lambda dest_acc, dest_sync, input_dimensions: get_valid_dest_indices(
        dest_sync=dest_sync,
        dest_acc=dest_acc,
        tile_count=(input_dimensions[0] * input_dimensions[1]) // (32 * 32),
    ),
)
def test_pack(
    formats,
    dest_acc,
    input_dimensions,
    relu_type,
    dest_sync,
    dest_index,
    workers_tensix_coordinates,
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

    configuration = TestConfig(
        "sources/pack_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            TILIZE(),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX(dest_index),
            RELU_CONFIG(relu_config),
            NUM_FACES(num_faces=4),
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

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor, res_tensor, formats.output_format, print_erros=False
    )

    if not test_passed and relu_type in [
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]:
        # When a datum is extremely close to the ReLU threshold, differences can arise due to
        # floating point precision limitations and rounding during format conversions.
        # We check if all mismatches are within a small tolerance of the threshold. If so, we consider the test as passed.
        is_tolerance_issue = is_relu_threshold_tolerance_issue(
            golden_tensor,
            res_tensor,
            relu_config,
            data_formats.pack_src,
        )

        if is_tolerance_issue:
            print(
                "Detected a packer ReLU threshold precision difference between hardware and software "
                "the discrepancy is within tolerance and is considered acceptable."
            )
            test_passed = True

    assert test_passed
