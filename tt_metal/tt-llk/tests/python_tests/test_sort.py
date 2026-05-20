# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLK regression for the metal `sort_single_row_single_core` algorithm
(local_sort + merge, no rebuild).

Per tile-row: first half = bf16 value tiles, second half = uint16 index
tiles encoding [0, Wt*32). Asserts result indices form a valid permutation
per row; duplicates / missing values (especially multiples of 256) signal
the tt-metal#37571 class of corruption.
"""

from collections import Counter

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TilizeGolden,
    TransposeGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, TopKSortDirection, format_dict
from helpers.logger import logger
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

NUM_STAGES = 2  # Values and Indices halves of the row.


def prepare_input_tensor_for_sort(src_A, formats, input_dimensions):
    """Identical to test_topk.prepare_input_tensor_for_topk.  Encodes
    arange(0, Wt*32) as uint16 in the index half of every row, then tilizes."""
    num_rows_tensor, num_cols_tensor = input_dimensions
    num_tiles_in_input = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

    if num_tiles_in_input < NUM_STAGES * 2:
        raise ValueError(
            f"Expected at least 4 tiles (2 value + 2 index), got {num_tiles_in_input}."
        )

    src_A = src_A.clone()
    half = num_cols_tensor // NUM_STAGES

    for row in range(num_rows_tensor):
        indices_start_idx = row * num_cols_tensor + half
        indices_end_idx = indices_start_idx + half

        uint16_indices = torch.arange(0, half, dtype=torch.int16).to(torch.uint16)
        src_A[indices_start_idx:indices_end_idx] = uint16_indices.view(src_A.dtype)

    src_tilizer = get_golden_generator(TilizeGolden)
    src_A = src_tilizer(src_A, input_dimensions, formats.input_format)
    return src_A


def transform_result_tensor_to_right_form(res_tensor, formats, input_dimensions):
    """Same approach as test_topk.transform_result_tensor_to_right_form, but
    operating on the FULL-sort output (Wt value tiles + Wt index tiles per row)."""
    num_rows_tensor, num_cols_tensor = input_dimensions
    res_tensor = res_tensor[0 : num_rows_tensor * num_cols_tensor]

    num_tiles = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

    transpose_util = get_golden_generator(TransposeGolden)
    res_tensor = transpose_util.transpose_faces_multi_tile(
        res_tensor,
        formats.output_format,
        num_tiles=num_tiles,
        tilize=False,
        untilize=False,
        input_dimensions=[num_rows_tensor, num_cols_tensor],
    )
    res_tensor = transpose_util.transpose_within_faces_multi_tile(
        res_tensor,
        formats.output_format,
        num_tiles=num_tiles,
        tilize=False,
        untilize=False,
        input_dimensions=[num_rows_tensor, num_cols_tensor],
    )
    return res_tensor


def validate_sort_permutation(
    res_tensor, original_input_tensor, formats, input_dimensions, atol=0.02
):
    """Validate the LLK sort output:
      1. indices form a valid permutation of [0, Wt*32) per row
      2. each result index points to the matching value in the original input

    On failure, log the corruption pattern (duplicates, missing values,
    multiples of 256 missing) so we can diagnose #37571.
    """
    num_rows_tensor, num_cols_tensor = input_dimensions
    half = num_cols_tensor // NUM_STAGES  # number of values per row
    indices_offset = half

    untilizer = get_golden_generator(UntilizeGolden)
    res_untilized = untilizer(res_tensor, formats.output_format, input_dimensions)
    original_untilized = untilizer(
        original_input_tensor, formats.input_format, input_dimensions
    )

    all_ok = True
    first_failure_row = None

    for row_idx in range(num_rows_tensor):
        # Extract uint16 indices for this row.
        row_indices = []
        row_values = []
        for k in range(half):
            value_idx = row_idx * num_cols_tensor + k
            index_idx = row_idx * num_cols_tensor + indices_offset + k
            row_values.append(res_untilized[value_idx].item())
            idx_uint16 = (
                res_untilized[index_idx : index_idx + 1].view(torch.uint16).item()
            )
            row_indices.append(idx_uint16)

        # Permutation check.
        idx_counter = Counter(row_indices)
        expected = set(range(half))
        present = set(row_indices)
        missing = sorted(expected - present)
        duplicates = sorted(
            [(v, c) for v, c in idx_counter.items() if c > 1], key=lambda x: x[0]
        )

        if missing or duplicates:
            all_ok = False
            if first_failure_row is None:
                first_failure_row = row_idx
                missing_mod_256 = [v for v in missing if v % 256 == 0]
                dup0_count = idx_counter.get(0, 0)
                logger.error(
                    "[#37571 reproducer] Row {} indices are NOT a valid permutation of [0, {}):\n"
                    "  missing count   = {}  (first 16: {})\n"
                    "  missing %256==0 = {}\n"
                    "  duplicates      = {}  (count of 0 = {})",
                    row_idx,
                    half,
                    len(missing),
                    missing[:16],
                    missing_mod_256[:16],
                    duplicates[:16],
                    dup0_count,
                )
            continue

        # Index -> value pointer check.
        for k in range(half):
            result_index = row_indices[k]
            result_value = row_values[k]
            original_value_at_idx = original_untilized[
                row_idx * num_cols_tensor + result_index
            ].item()
            if not torch.isclose(
                torch.tensor(result_value),
                torch.tensor(original_value_at_idx),
                atol=atol,
            ):
                if first_failure_row is None:
                    first_failure_row = row_idx
                    logger.error(
                        "[#37571 reproducer] Row {}, datum {}: index {} "
                        "points to {} in input, but result has {}",
                        row_idx,
                        k,
                        result_index,
                        original_value_at_idx,
                        result_value,
                    )
                all_ok = False
                break

    return all_ok


def _wt_to_input_dim(Wt: int) -> list:
    """Wt is the number of value tiles per row.  Total tiles per row is
    2*Wt (Wt values + Wt indices), so the per-row column count is 2*Wt*32.
    We only test a single tile-row for speed."""
    return [32, 2 * Wt * 32]


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    Wt=[8, 16, 32],
)
def test_sort_sfpu(formats: InputOutputFormat, Wt: int):
    """Reproducer for tt-metal issue #37571.

    Wt = 8  -> 256 elements per row -> EXPECTED TO PASS (no merge stages, only local_sort).
    Wt = 16 -> 512 elements per row -> EXPECTED TO FAIL (4-stage merge, the broken path).
    Wt = 32 -> 1024 elements per row -> EXPECTED TO FAIL (5-stage merge, sibling failure).
    """
    input_dimensions = _wt_to_input_dim(Wt)
    K = Wt * 32  # for the TOPK template params: makes log2(K) = log2(Wt*32)
    # so TOPK_NUM_ITERATIONS = log2(Wt) - matches our STAGES.
    sort_direction = TopKSortDirection.Ascending

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    src_A = prepare_input_tensor_for_sort(src_A, formats, input_dimensions)

    configuration = TestConfig(
        test_name="sources/sort_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=K,
                topk_matrix_width=input_dimensions[1],
                topk_sort_direction=sort_direction,
                topk_stable_sort=False,
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
        res_tensor, formats, input_dimensions
    )

    assert len(res_tensor) == input_dimensions[0] * input_dimensions[1], (
        f"Result tensor length {len(res_tensor)} != expected "
        f"{input_dimensions[0] * input_dimensions[1]}"
    )

    ok = validate_sort_permutation(res_tensor, src_A, formats, input_dimensions)
    assert ok, (
        f"Sort permutation invalid for Wt={Wt} (input_dimensions={input_dimensions}). "
        "See stderr for the corruption pattern."
    )
