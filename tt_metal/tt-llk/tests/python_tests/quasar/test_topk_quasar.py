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
    TransposeGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, TopKSortDirection, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    INPUT_DIMENSIONS,
    TILE_COUNT,
    TOPK,
)
from helpers.unpack import unpack_res_tiles
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import read_from_device, write_to_device

NUM_STAGES = 2  # Values and Indices stage
TOPK_INDEX_FORMAT = DataFormat.Int16


def is_index_tile(tile_index: int, value_tiles_per_row: int) -> bool:
    tile_col = tile_index % (value_tiles_per_row * NUM_STAGES)
    return tile_col >= value_tiles_per_row


class TopKStimuliConfig(StimuliConfig):
    def __init__(
        self,
        *args,
        input_value_tiles_per_row: int,
        result_value_tiles_per_row: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_value_tiles_per_row = input_value_tiles_per_row
        self.result_value_tiles_per_row = result_value_tiles_per_row

    def _write_mixed_topk_input_A(self, location: str):
        for tile_index in range(self.tile_count_A):
            tile_format = (
                TOPK_INDEX_FORMAT
                if is_index_tile(tile_index, self.input_value_tiles_per_row)
                else self.stimuli_A_format
            )
            pack_function = StimuliConfig.get_packer(tile_format)
            if not pack_function:
                raise ValueError(f"Unsupported TopK tile format: {tile_format.name}")

            start_idx = ELEMENTS_PER_TILE * tile_index
            tile_data = self.buffer_A[start_idx : start_idx + ELEMENTS_PER_TILE].to(
                format_dict[tile_format]
            )
            packed_data = pack_function(tile_data)
            write_to_device(
                location,
                self.buf_a_addr + tile_index * self.tile_size_A_bytes,
                packed_data,
            )

    def _write_backward_compatible(self, location: str = "0,0"):
        self._write_mixed_topk_input_A(location)

        pack_function_B = StimuliConfig.get_packer(self.stimuli_B_format)
        if not pack_function_B:
            raise ValueError(
                f"Unsupported data format for operand B: {self.stimuli_B_format.name}"
            )

        StimuliConfig.write_matrix(
            self.buffer_B,
            self.tile_count_B,
            pack_function_B,
            self.buf_b_addr,
            self.tile_size_B_bytes,
            self.num_faces,
            self.face_r_dim,
            location,
            self.write_full_tiles,
            use_srcs=self.use_srcs,
            twos_complement=self.twos_complement,
        )

    def collect_results(self, location="0,0"):
        read_bytes_cnt = self.buf_res_tile_size * self.tile_count_res
        read_data = read_from_device(
            location, self.buf_res_addr, num_bytes=read_bytes_cnt
        )

        unpacked_data = []
        for tile_index in range(self.tile_count_res):
            tile_format = (
                TOPK_INDEX_FORMAT
                if is_index_tile(tile_index, self.result_value_tiles_per_row)
                else self.stimuli_res_format
            )
            start_idx = tile_index * self.buf_res_tile_size
            end_idx = start_idx + self.buf_res_tile_size
            tile_data = read_data[start_idx:end_idx]
            unpacked_tile = unpack_res_tiles(
                tile_data,
                tile_format,
                tile_count=1,
                sfpu=self.sfpu,
                num_faces=self.num_faces,
                face_r_dim=self.face_r_dim,
                tile_stride_bytes=self.buf_res_tile_size,
                use_srcs=self.use_srcs,
                dest_acc=self._dest_acc_32b,
                twos_complement=self.twos_complement,
            )
            unpacked_data.extend(unpacked_tile.to(torch.float32).tolist())

        return torch.tensor(unpacked_data, dtype=torch.float32)


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
    values_per_row = num_cols_tensor // NUM_STAGES
    num_tiles_in_input = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

    if num_tiles_in_input < NUM_STAGES * 2:
        raise ValueError(
            f"Expected at least 2 tiles for values and 2 tiles for indices (total 4 tiles), but got {num_tiles_in_input} tiles."
        )

    src_A = src_A.to(torch.float32).clone().view(num_rows_tensor, num_cols_tensor)

    for row in range(num_rows_tensor):
        src_A[row, values_per_row:] = torch.arange(values_per_row, dtype=torch.float32)

    src_tilizer = get_golden_generator(TilizeGolden)
    value_tiles = (
        src_tilizer(
            src_A[:, :values_per_row].flatten(),
            [num_rows_tensor, values_per_row],
            formats.input_format,
        )
        .to(torch.float32)
        .view(-1, ELEMENTS_PER_TILE)
    )
    index_tiles = (
        src_tilizer(
            src_A[:, values_per_row:].flatten(),
            [num_rows_tensor, values_per_row],
            TOPK_INDEX_FORMAT,
        )
        .to(torch.float32)
        .view(-1, ELEMENTS_PER_TILE)
    )

    value_tiles_per_row = values_per_row // TILE_DIMENSIONS[1]
    tile_rows = num_rows_tensor // TILE_DIMENSIONS[0]
    mixed_tiles = []
    for tile_row in range(tile_rows):
        row_start = tile_row * value_tiles_per_row
        row_end = row_start + value_tiles_per_row
        mixed_tiles.extend(value_tiles[row_start:row_end])
        mixed_tiles.extend(index_tiles[row_start:row_end])

    return torch.cat(mixed_tiles)


def generate_topk_golden_with_numeric_indices(
    src_A,
    formats,
    K=32,
    sort_direction=TopKSortDirection.Descending,
    input_dimensions=[32, 128],
):
    num_rows_tensor, num_cols_tensor = input_dimensions
    result_num_cols = NUM_STAGES * K

    operand = src_A.to(format_dict[formats.input_format])
    result = torch.zeros(num_rows_tensor * result_num_cols, dtype=torch.float32)

    for row in range(num_rows_tensor):
        values_start_idx = row * num_cols_tensor
        values_end_idx = values_start_idx + num_cols_tensor // NUM_STAGES
        values = operand[values_start_idx:values_end_idx]

        topk_positions = torch.argsort(
            values,
            descending=(sort_direction == TopKSortDirection.Descending),
            stable=True,
        )[:K]
        topk_values = values[topk_positions]

        result_values_start_idx = row * result_num_cols
        result_indices_start_idx = (
            result_values_start_idx + result_num_cols // NUM_STAGES
        )

        result[result_values_start_idx : result_values_start_idx + K] = topk_values.to(
            torch.float32
        )
        result[result_indices_start_idx : result_indices_start_idx + K] = (
            topk_positions.to(torch.float32)
        )

    result_tilizer = get_golden_generator(TilizeGolden)
    result = result_tilizer(
        result,
        dimensions=[num_rows_tensor, result_num_cols],
        data_format=formats.output_format,
    )

    return result.to(torch.float32)


def make_unique_value_input(src_A, input_dimensions=[32, 128]):
    src_A = src_A.to(torch.float32).clone()
    num_rows_tensor, num_cols_tensor = input_dimensions
    values_per_row = num_cols_tensor // NUM_STAGES

    unique_values = torch.arange(values_per_row, dtype=torch.float32)
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
    ).to(torch.float32)
    golden_tensor_untilized = untilizer(
        golden_tensor, formats.output_format, [num_rows_tensor, num_cols_tensor]
    ).to(torch.float32)
    original_input_tensor_untilized = untilizer(
        original_input_tensor, formats.input_format, input_dimensions
    ).to(torch.float32)

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

            result_index = int(
                round(float(res_tensor_untilized[result_and_golden_index_idx].item()))
            )
            golden_index = int(
                round(
                    float(golden_tensor_untilized[result_and_golden_index_idx].item())
                )
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
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if unique_values:
        src_A = make_unique_value_input(src_A, input_dimensions)

    golden_tensor = generate_topk_golden_with_numeric_indices(
        src_A,
        formats,
        K,
        sort_direction,
        input_dimensions=input_dimensions,
    )

    src_A = prepare_input_tensor_for_topk(src_A, formats, input_dimensions)

    input_value_tiles_per_row = input_dimensions[1] // TILE_DIMENSIONS[1] // NUM_STAGES
    result_value_tiles_per_row = K // TILE_DIMENSIONS[1]

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
        variant_stimuli=TopKStimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            input_value_tiles_per_row=input_value_tiles_per_row,
            result_value_tiles_per_row=result_value_tiles_per_row,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=torch.float32)

    res_tensor = transform_result_tensor_to_right_form(
        res_tensor, formats, K, input_dimensions
    )

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

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

    if input_dimensions == [32, 1024]:
        pytest.skip(
            "Skipping 32x1024 TopK on Quasar due to observed index/value mismatches in the multi-iteration path."
        )

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
