# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TopK SFPU Test for Quasar Architecture

Tests the hardware TopK operation using iterative bitonic merge algorithm
on the Quasar simulator. Validates extraction of top K values from input
tensors across multiple rows, verifying both sorted values and corresponding
index tracking.

This is a Quasar-specific wrapper around the shared topk test infrastructure.
It uses the shared C++ test source (sources/topk_test.cpp) which already has
ARCH_BLACKHOLE / else branches for Quasar support.

Input Layout:
- First half of columns: Value tiles to search for top K elements
- Second half of columns: Index tiles (integer format) tracking original positions

Algorithm:
- Processes each row independently through TOPK_NUM_ITERATIONS of pairwise merges
- First iteration transposes to column-major and performs local sort
- Subsequent iterations merge sorted pairs, halving tile count each time
- Final output contains K values and K indices per row in specified sort order
"""

import sys

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIMENSIONS,
    TilizeGolden,
    TopKGolden,
    TransposeGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, TopKSortDirection, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    INPUT_DIMENSIONS,
    TILE_COUNT,
    TOPK,
)
from helpers.utils import _RECORD_TEST_ORDER, passed_test

NUM_STAGES = 2  # Values and Indices stage


def transform_result_tensor_to_right_form(
    res_tensor, formats, K=32, input_dimensions=[32, 64]
):
    # Cut the result tensor to the actual expected golden size. Ignore the rest.
    num_rows_tensor, num_cols_tensor = (
        input_dimensions[0],
        K * NUM_STAGES,
    )  # K values + K indices

    res_tensor = res_tensor[0 : num_rows_tensor * num_cols_tensor]

    num_tiles_in_input = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

    if num_tiles_in_input < NUM_STAGES:
        raise ValueError(
            f"Expected at least 1 tile for values and 1 tile for indices (total 2 tiles), but got {num_tiles_in_input} tiles."
        )

    # We need to transpose the result to return it to the original row-wise order.
    transpose_util = get_golden_generator(TransposeGolden)

    # First: transpose faces (swap face positions).
    res_tensor = transpose_util.transpose_faces_multi_tile(
        res_tensor,
        formats.output_format,
        num_tiles=num_tiles_in_input,
        tilize=False,
        untilize=False,
        input_dimensions=[num_rows_tensor, num_cols_tensor],
    )

    # Then: transpose within each face.
    res_tensor = transpose_util.transpose_within_faces_multi_tile(
        res_tensor,
        formats.output_format,
        num_tiles=num_tiles_in_input,
        tilize=False,
        untilize=False,
        input_dimensions=[num_rows_tensor, num_cols_tensor],
    )

    return res_tensor


def prepare_input_tensor_for_topk(src_A, formats, input_dimensions=[32, 128]):
    num_rows_tensor, num_cols_tensor = input_dimensions
    num_tiles_in_input = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

    if num_tiles_in_input < NUM_STAGES * 2:
        raise ValueError(
            f"Expected at least 2 tiles for values and 2 tiles for indices (total 4 tiles), but got {num_tiles_in_input} tiles."
        )

    # Clone to avoid modifying the original tensor.
    src_A = src_A.clone()

    # These will be used as indices for the topk operation, and we want them to be in a known order for easier validation.
    # Create indices as uint16 and preserve bit representation when assigning to float tensor.
    for row in range(num_rows_tensor):
        indices_start_idx = row * num_cols_tensor + num_cols_tensor // NUM_STAGES
        indices_end_idx = indices_start_idx + num_cols_tensor // NUM_STAGES

        uint16_indices = torch.arange(
            0, num_cols_tensor // NUM_STAGES, dtype=torch.int16
        ).to(torch.uint16)

        src_A[indices_start_idx:indices_end_idx] = uint16_indices.view(src_A.dtype)

    src_tilizer = get_golden_generator(TilizeGolden)
    src_A = src_tilizer(src_A, input_dimensions, formats.input_format)

    return src_A


def make_unique_value_input(src_A, input_dimensions=[32, 128]):
    src_A = src_A.clone()
    num_rows_tensor, num_cols_tensor = input_dimensions
    values_per_row = num_cols_tensor // NUM_STAGES

    unique_values = torch.arange(values_per_row, dtype=torch.float32).to(src_A.dtype)
    for row in range(num_rows_tensor):
        values_start_idx = row * num_cols_tensor
        values_end_idx = values_start_idx + values_per_row
        src_A[values_start_idx:values_end_idx] = unique_values

    return src_A


def validate_topk_indices(
    res_tensor,
    golden_tensor,
    original_input_tensor,
    formats,
    input_dimensions=[32, 128],
    K=32,
    stable_sort=False,
    atol=0.01,
):
    num_rows_tensor, num_cols_tensor = (
        input_dimensions[0],
        K * NUM_STAGES,
    )  # K values + K indices
    num_tiles_in_input = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

    if num_tiles_in_input < NUM_STAGES:
        raise ValueError(
            f"Expected at least 1 tile for values and 1 tile for indices (total 2 tiles), but got {num_tiles_in_input} tiles."
        )

    # Untilize both result and golden tensors to get them back to the original layout for easier comparison.
    untilizer = get_golden_generator(UntilizeGolden)
    res_tensor_untilized = untilizer(
        res_tensor, formats.output_format, [num_rows_tensor, num_cols_tensor]
    )
    golden_tensor_untilized = untilizer(
        golden_tensor, formats.output_format, [num_rows_tensor, num_cols_tensor]
    )
    original_input_tensor_untilized = untilizer(
        original_input_tensor, formats.input_format, input_dimensions
    )

    values_offset = 0
    indices_offset = num_cols_tensor // 2  # Indices stored in second half of row.

    for row_idx in range(input_dimensions[0]):
        for datum in range(K):  # Check top K values/indices for each row.
            result_and_golden_value_idx = (
                row_idx * num_cols_tensor + values_offset + datum
            )
            result_and_golden_index_idx = (
                row_idx * num_cols_tensor + indices_offset + datum
            )

            # Values: interpret as float
            result_value = res_tensor_untilized[result_and_golden_value_idx].item()
            golden_value = golden_tensor_untilized[result_and_golden_value_idx].item()

            # Indices: reinterpret float bits as uint16 as that's how we encoded them in the input tensor.
            result_index = (
                res_tensor_untilized[
                    result_and_golden_index_idx : result_and_golden_index_idx + 1
                ]
                .view(torch.uint16)
                .item()
            )
            golden_index = (
                golden_tensor_untilized[
                    result_and_golden_index_idx : result_and_golden_index_idx + 1
                ]
                .view(torch.uint16)
                .item()
            )

            original_input_value_idx = row_idx * input_dimensions[1] + result_index
            original_input_value = original_input_tensor_untilized[
                original_input_value_idx
            ].item()

            # Check if the result index actually points to the same value in the result tensor as in the input tensor.
            if result_value != original_input_value:
                print(
                    f"Index-value mismatch at row {row_idx}, datum {datum}:",
                    file=sys.stderr,
                )
                print(
                    f"  Result value: {result_value} with index {result_index} does not match original input value: {original_input_value} at the same index.",
                    file=sys.stderr,
                )
                return False

            if result_index != golden_index:
                if (
                    torch.isclose(
                        torch.tensor(result_value),
                        torch.tensor(golden_value),
                        atol=atol,
                    )
                    and stable_sort is False
                ):
                    # When doing topk with unstable sort, we can encounter cases where the values are extremely close/same.
                    # In those cases golden has its own way of deciding which index to pick first, and hardware might pick a different one.
                    # What we get in the end is that the same values are in the topk, but maybe in a different order, which means different indices.
                    # This is not an issue, just the difference between golden and hardware when handling ties in values.
                    continue
                else:
                    print(f"Mismatch at row {row_idx}, datum {datum}:", file=sys.stderr)
                    print(
                        f"  Result value: {result_value}, Result index: {result_index}",
                        file=sys.stderr,
                    )
                    print(
                        f"  Golden value: {golden_value}, Golden index: {golden_index}",
                        file=sys.stderr,
                    )
                    return False
    return True


def get_value_tiles_from_topk_tensor(
    tensor: torch.Tensor, K: int = 32, input_dimensions=[32, 128]
):
    # Get the value tiles from the topk result tensor. This is useful for validating the topk values separately from the indices,
    # since indices can differ in tie cases but values should still match.
    num_rows, num_cols = input_dimensions[0], K * NUM_STAGES  # K values + K indices
    num_tile_rows = num_rows // TILE_DIMENSIONS[0]
    num_tile_cols = num_cols // TILE_DIMENSIONS[1]
    num_value_tiles_per_row = (
        K // TILE_DIMENSIONS[1]
    )  # Number of tiles that contain the top K values in each row.

    tiles = []

    for tile_row in range(num_tile_rows):
        for tile_col in range(num_value_tiles_per_row):
            # In tilized format, tiles are stored in row-major order
            tile_index = tile_row * num_tile_cols + tile_col
            start_idx = tile_index * ELEMENTS_PER_TILE
            end_idx = start_idx + ELEMENTS_PER_TILE
            tiles.append(tensor[start_idx:end_idx])

    return torch.cat(tiles)


def run_topk_quasar_case(
    formats: InputOutputFormat,
    input_dimensions: list,
    K: int,
    sort_direction: TopKSortDirection,
    stable_sort: bool,
    unique_values: bool = False,
):
    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    if unique_values:
        src_A = make_unique_value_input(src_A, input_dimensions)

    golden_generator = get_golden_generator(TopKGolden)
    golden_tensor = golden_generator(
        src_A,
        formats.input_format,
        K,
        sort_direction,
        input_dimensions=input_dimensions,
    )

    src_A = prepare_input_tensor_for_topk(src_A, formats, input_dimensions)

    configuration = TestConfig(
        test_name="sources/quasar/sfpu_topk_quasar_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=K,
                topk_matrix_width=input_dimensions[1],
                topk_sort_direction=sort_direction,
                topk_stable_sort=stable_sort,
            ),
        ],
        runtimes=[
            INPUT_DIMENSIONS(input_dimensions[0] // 32, input_dimensions[1] // 32),
            TILE_COUNT(tile_cnt_A),
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
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    res_tensor = transform_result_tensor_to_right_form(
        res_tensor, formats, K, input_dimensions
    )

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    if input_dimensions[1] == 128 and not _RECORD_TEST_ORDER:
        assert validate_topk_indices(
            res_tensor, golden_tensor, src_A, formats, input_dimensions, K, stable_sort
        )

    res_values = get_value_tiles_from_topk_tensor(res_tensor, K, input_dimensions)
    golden_values = get_value_tiles_from_topk_tensor(golden_tensor, K, input_dimensions)

    assert passed_test(
        golden_values, res_values, formats.output_format, print_errors=True
    )


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    input_dimensions=[
        [32, 128],
        [64, 128],
        [256, 128],
        [32, 256],
        [32, 512],
        [32, 1024],
    ],
    K=[32],
    sort_direction=[TopKSortDirection.Descending, TopKSortDirection.Ascending],
    stable_sort=[False, True],
)
def test_topk_quasar(
    formats: InputOutputFormat,
    input_dimensions: list,
    K: int,
    sort_direction: TopKSortDirection,
    stable_sort: bool,
):
    if stable_sort:
        pytest.skip(
            "Stable sort is currently broken in LLK API."
        )  # TODO: Check tenstorrent/tt-metal#33492 and remove this once fixed.

    run_topk_quasar_case(
        formats,
        input_dimensions,
        K,
        sort_direction,
        stable_sort,
    )


@pytest.mark.quasar
def test_topk_quasar_unique_indices():
    run_topk_quasar_case(
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        [32, 128],
        32,
        TopKSortDirection.Descending,
        False,
        unique_values=True,
    )
