# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TopK SFPU Test

Tests the hardware TopK operation using iterative bitonic merge algorithm.
Validates extraction of top K values from input tensors across multiple rows,
verifying both sorted values and corresponding index tracking.

Input Layout:
- First half of columns: Value tiles to search for top K elements
- Second half of columns: Index tiles (integer format) tracking original positions

Algorithm:
- Processes each row independently through TOPK_NUM_ITERATIONS of pairwise merges
- First iteration transposes to column-major and performs local sort
- Subsequent iterations merge sorted pairs, halving tile count each time
- Final output contains K values and K indices per row in specified sort order

Validation:
- Compares hardware results against PyTorch topk golden reference
- Handles tie-breaking differences between hardware and PyTorch
# Validates both value accuracy and index correctness
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

    # Untilize both result and golden tensors to get them back to the original layout for easier(cleaner) comparison
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
                    # in those cases golden has its own way of deciding which index to pick first, and hardware might pick a different one.
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
    # Wider dims (W >= 256) and K=64 are blocked on the wide-width harness
    # discrepancies tracked in tt-llk#1344; sub-tile K and K=64 coverage lives
    # in the ttnn-level topk tests.
    K=[32],
    sort_direction=[TopKSortDirection.Descending, TopKSortDirection.Ascending],
    # unstable / comparator-stable / fused-key-stable (packed [bf16|u16] keys, unstable
    # network) / rank-stamped-stable (sign-conditioned local-rank tags in the value lo16,
    # true indices riding index tracking, unstable network).
    sort_mode=["unstable", "stable", "fused", "rank_stamped"],
)
def test_topk_sfpu(
    formats: InputOutputFormat,
    input_dimensions: list,
    K: int,
    sort_direction: TopKSortDirection,
    sort_mode: str,
):
    stable_sort = sort_mode == "stable"
    fused_stable = sort_mode == "fused"
    rank_stamped = sort_mode == "rank_stamped"

    if input_dimensions == [32, 1024]:
        # For 32x1024 input we have observed some discrepancies in the topk values between hardware and golden.
        # TODO: Fix issue #1344 on tt-llk.
        pytest.skip("Skipping test for 32x1024 input due to observed discrepancies.")

    if fused_stable and input_dimensions[1] != 128:
        # The fused kernel path handles a single 2-tile slab per pipeline (TOPK_NUM_ITERATIONS == 1);
        # multi-iteration fused slabs need the packed-CB round-trip that lands with the ttnn milestone.
        pytest.skip(
            "Fused stable mode currently covers single-iteration widths (W == 128) only."
        )

    if rank_stamped and input_dimensions[1] != 128:
        # Rank tags do not survive this harness's bf16 L1 round-trip between iterations (the ttnn
        # pipeline re-stamps inside the merge and moves value words through raw Float32 CBs).
        pytest.skip(
            "Rank-stamped stable mode covers single-iteration widths (W == 128) in this harness."
        )

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

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
        test_name="sources/topk_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=K,
                topk_matrix_width=input_dimensions[1],
                topk_sort_direction=sort_direction,
                topk_stable_sort=stable_sort,
                topk_fused_stable=fused_stable,
                topk_rank_stamped=rank_stamped,
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
        # Fused / rank-stamped keys are 32-bit words: values must be exact-widened into 32-bit DEST.
        dest_acc=(
            DestAccumulation.Yes
            if (fused_stable or rank_stamped)
            else DestAccumulation.No
        ),
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

    # TODO: Fix issue #1344 on tt-llk.
    if input_dimensions[1] == 128 and not _RECORD_TEST_ORDER:
        # Fused and rank-stamped modes promise the same torch-stable tie order as
        # comparator-stable, so they get the strict (no tie-escape) index comparison too.
        assert validate_topk_indices(
            res_tensor,
            golden_tensor,
            src_A,
            formats,
            input_dimensions,
            K,
            stable_sort or fused_stable or rank_stamped,
        )

    # Get value tiles from result and golden tensors
    res_values = get_value_tiles_from_topk_tensor(res_tensor, K, input_dimensions)
    golden_values = get_value_tiles_from_topk_tensor(golden_tensor, K, input_dimensions)

    # Validate topk values
    assert passed_test(
        golden_values, res_values, formats.output_format, print_errors=True
    )


# =============================================================================
# Adversarial stimuli coverage (test_topk_sfpu_adversarial)
# =============================================================================
#
# Bit-pattern stimuli (tie pileups, signed zeros, bf16 denormals, infinities, NaN
# payloads) that the uniform stimuli above cannot produce, checked per row against
# an in-test golden. Both engines canonicalize before sorting: +-0 and denormals
# -> +0.0, NaN -> same-sign infinity bits, normals and +-inf kept, ties broken
# index-ascending (measured on Blackhole silicon).

_BF16_SIGN_MASK = 0x8000
_BF16_EXP_MASK = 0x7F80
_BF16_MANT_MASK = 0x007F

_BF16_POS_INF = 0x7F80
_BF16_NEG_INF = 0xFF80
_BF16_POS_ZERO = 0x0000
_BF16_NEG_ZERO = 0x8000
_BF16_QNAN = 0x7FC0
_BF16_SNAN_PAYLOAD = 0x7F81  # Quietized to 0x7FC0 by the harness write path.
_BF16_NEG_QNAN = 0xFFC0
_BF16_MIN_DENORM = 0x0001
_BF16_MAX_DENORM = 0x007F
_BF16_NEG_MIN_DENORM = 0x8001

ADVERSARIAL_STIMULI_CLASSES = [
    "neg_ties",
    "mixed_sign_ties",
    "signed_zero",
    "infinities",
    "nan_payloads",
    "tie_straddle_k",
]

_ADVERSARIAL_CLASS_SEED = {
    name: 0xAD00 + 257 * i for i, name in enumerate(ADVERSARIAL_STIMULI_CLASSES)
}


def _bf16_bits_from_floats(values):
    """bf16 bit patterns (uint16) of a list of exactly-representable floats."""
    return torch.tensor(values, dtype=torch.bfloat16).view(torch.uint16)


def _gather_u16(bits_u16, order):
    """Advanced indexing routed through an int16 view (bit-preserving): CPU
    torch does not implement tensor indexing for uint16."""
    return bits_u16.view(torch.int16)[order].view(torch.uint16)


def _is_nan_bits(bits_i32):
    return ((bits_i32 & _BF16_EXP_MASK) == _BF16_EXP_MASK) & (
        (bits_i32 & _BF16_MANT_MASK) != 0
    )


def _canon_bits(bits_u16):
    """Model of the engines' pre-sort canonicalization (ttnn-level silicon):
    exp==0 patterns (+-0 and all bf16 denormals) -> +0.0 (0x0000); NaN ->
    same-sign infinity bits (tying genuinely with real same-sign inf); normals
    and +-inf unchanged."""
    b = bits_u16.to(torch.int32)
    b = torch.where((b & _BF16_EXP_MASK) == 0, torch.zeros_like(b), b)
    b = torch.where(_is_nan_bits(b), (b & _BF16_SIGN_MASK) | _BF16_POS_INF, b)
    return b.to(torch.uint16)


def _model_write_path_bits(bits_u16):
    """Harness L1 write path (pack_bfp16): NaNs land in L1 as the sign-preserved
    canonical qNaN (payload lost); everything else is written bit-exactly."""
    b = bits_u16.to(torch.int32)
    b = torch.where(_is_nan_bits(b), (b & _BF16_SIGN_MASK) | _BF16_QNAN, b)
    return b.to(torch.uint16)


def _canon_stable_golden(row_bits_u16, K, descending):
    """(expected_value_bits, expected_indices) under the canonicalizing-engine
    hypothesis: stable torch argsort (index-ascending ties) over the
    canonicalized values. Canonicalized outputs are normals / +0 / +-inf, all
    preserved bit-exactly by the readback path, so the expected value bits
    need no further transport modeling."""
    canon = _canon_bits(row_bits_u16)
    values = canon.view(torch.bfloat16).to(torch.float32)
    order = torch.argsort(values, descending=descending, stable=True)[:K]
    return _gather_u16(canon, order), order


def _adversarial_value_bits(stimuli_class, num_rows, w_values):
    """Per-row bf16 bit patterns (uint16, [num_rows, w_values]) for one
    adversarial class. Every row holds the same multiset in a deterministically
    different order (generator seeded per class+row), so the stimulus is
    reproducible run to run."""
    if stimuli_class == "neg_ties":
        # 6 negative tie levels; repeats sized to fill the row.
        levels = [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0]
        counts = [11, 11, 11, 11, 10, 10]
        base = _bf16_bits_from_floats(
            [level for level, count in zip(levels, counts) for _ in range(count)]
        )
    elif stimuli_class == "mixed_sign_ties":
        # +/- tie levels interleaved so opposite-sign ties sit adjacent.
        levels = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0]
        base = _bf16_bits_from_floats(levels * (w_values // len(levels)))
    elif stimuli_class == "signed_zero":
        specials = (
            [_BF16_POS_ZERO] * 13
            + [_BF16_NEG_ZERO] * 13
            + [_BF16_MIN_DENORM] * 2
            + [_BF16_MAX_DENORM] * 2
            + [_BF16_NEG_MIN_DENORM] * 2
        )
        # 16 distinct negative + 16 distinct positive normal fillers, so the
        # 32-strong canon-zero group straddles K in both sort directions.
        fillers = [-1.0 - 0.25 * i for i in range(16)] + [
            1.0 + 0.25 * i for i in range(16)
        ]
        base = torch.cat(
            [
                torch.tensor(specials, dtype=torch.uint16),
                _bf16_bits_from_floats(fillers),
            ]
        )
    elif stimuli_class == "infinities":
        specials = [_BF16_POS_INF] * 6 + [_BF16_NEG_INF] * 6
        # 52 distinct finite fillers straddling zero without producing 0.0.
        fillers = [(i - 26) * 0.25 + 0.125 for i in range(52)]
        base = torch.cat(
            [
                torch.tensor(specials, dtype=torch.uint16),
                _bf16_bits_from_floats(fillers),
            ]
        )
    elif stimuli_class == "nan_payloads":
        specials = [_BF16_QNAN] * 2 + [_BF16_SNAN_PAYLOAD] * 2 + [_BF16_NEG_QNAN] * 2
        # 58 distinct finite fillers straddling zero without producing 0.0.
        fillers = [(i - 29) * 0.25 + 0.125 for i in range(58)]
        base = torch.cat(
            [
                torch.tensor(specials, dtype=torch.uint16),
                _bf16_bits_from_floats(fillers),
            ]
        )
    elif stimuli_class == "tie_straddle_k":
        # 48 copies of one level; with K=32 the tie group straddles the cut in
        # both sort directions (8 distinct fillers above, 8 below the level).
        base = torch.cat(
            [
                _bf16_bits_from_floats([1.0] * 48),
                _bf16_bits_from_floats([0.25 + 0.03125 * i for i in range(8)]),
                _bf16_bits_from_floats([2.0 + 0.125 * i for i in range(8)]),
            ]
        )
    else:
        raise ValueError(f"Unknown adversarial stimuli class: {stimuli_class}")

    if base.numel() != w_values:
        raise ValueError(
            f"Class '{stimuli_class}' builds {base.numel()} values per row, "
            f"expected {w_values}."
        )

    rows = []
    for row in range(num_rows):
        if stimuli_class == "mixed_sign_ties":
            # Rotate instead of shuffling to keep the +/- interleaving intact.
            perm = (torch.arange(w_values) + row) % w_values
        else:
            generator = torch.Generator()
            generator.manual_seed(_ADVERSARIAL_CLASS_SEED[stimuli_class] + row)
            perm = torch.randperm(w_values, generator=generator)
        rows.append(_gather_u16(base, perm))
    return torch.stack(rows)


def _extract_topk_values_and_indices(res_tensor, formats, num_rows, K):
    """Split the (transformed, tilized) device result into per-row halves.
    Mirrors validate_topk_indices's extraction: values are the first K columns
    of each row, indices the second K columns read as uint16 bit patterns."""
    untilizer = get_golden_generator(UntilizeGolden)
    untilized = untilizer(res_tensor, formats.output_format, [num_rows, K * NUM_STAGES])
    rows2d = untilized.reshape(num_rows, K * NUM_STAGES)
    value_bits = rows2d[:, :K].contiguous().view(torch.uint16)
    indices = rows2d[:, K:].contiguous().view(torch.uint16).to(torch.int64)
    return value_bits, indices


def _hex_row(bits_u16):
    return "[" + " ".join(f"{v:04X}" for v in bits_u16.tolist()) + "]"


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    # Single 2-tile slab per stage; also satisfies the fused-mode W == 128
    # constraint, so no mode is skipped.
    input_dimensions=[[32, 128]],
    K=[32],
    sort_direction=[TopKSortDirection.Descending, TopKSortDirection.Ascending],
    sort_mode=["stable", "fused", "rank_stamped"],
    stimuli_class=ADVERSARIAL_STIMULI_CLASSES,
)
def test_topk_sfpu_adversarial(
    formats: InputOutputFormat,
    input_dimensions: list,
    K: int,
    sort_direction: TopKSortDirection,
    sort_mode: str,
    stimuli_class: str,
):
    """
    Adversarial bit-pattern coverage for the stable (comparator) and fused
    topk engines.

    Each stimuli class fills the value half of every row with raw bf16 bit
    patterns and validates result indices and value bits per row against
    in-test goldens (see the adversarial-coverage comment block above).
    The unstable engine is deliberately excluded: with heavy ties its output
    order is unspecified, so no exact golden exists for these classes.
    """
    stable_sort = sort_mode == "stable"
    fused_stable = sort_mode == "fused"
    rank_stamped = sort_mode == "rank_stamped"
    descending = sort_direction == TopKSortDirection.Descending

    # Per-test seed for anything drawing from the global RNG (e.g. src_B);
    # the value halves use dedicated per-class+row generators.
    torch.manual_seed(0)

    num_rows, num_cols = input_dimensions
    w_values = num_cols // NUM_STAGES

    # src_B and the tile counts come from the standard generator; the value
    # half of src_A is rebuilt from raw bit patterns below.
    filler_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=filler_spec,
        spec_B=filler_spec,
    )

    row_bits = _adversarial_value_bits(stimuli_class, num_rows, w_values)

    src_A = src_A.clone().view(num_rows, num_cols)
    src_A[:, :w_values] = row_bits.view(torch.bfloat16)
    src_A = src_A.flatten()

    # Reuse the harness preparation for the index half (u16 iota tiles).
    src_A = prepare_input_tensor_for_topk(src_A, formats, input_dimensions)

    configuration = TestConfig(
        test_name="sources/topk_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=K,
                topk_matrix_width=input_dimensions[1],
                topk_sort_direction=sort_direction,
                topk_stable_sort=stable_sort,
                topk_fused_stable=fused_stable,
                topk_rank_stamped=rank_stamped,
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
        # Fused / rank-stamped keys are 32-bit words: values must be exact-widened into 32-bit DEST.
        dest_acc=(
            DestAccumulation.Yes
            if (fused_stable or rank_stamped)
            else DestAccumulation.No
        ),
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    res_tensor = transform_result_tensor_to_right_form(
        res_tensor, formats, K, input_dimensions
    )

    if _RECORD_TEST_ORDER:
        # Order-recording pass: results are not meaningful for strict checks.
        return

    res_value_bits, res_indices = _extract_topk_values_and_indices(
        res_tensor, formats, num_rows, K
    )

    # Goldens are computed on the bits the device actually received (the write
    # path quietizes sNaN payloads before they reach L1).
    as_written_bits = _model_write_path_bits(row_bits)

    for row in range(num_rows):
        exp_bits, exp_idx = _canon_stable_golden(as_written_bits[row], K, descending)
        if torch.equal(res_indices[row], exp_idx) and torch.equal(
            res_value_bits[row], exp_bits
        ):
            continue
        pytest.fail(
            "\n".join(
                [
                    f"topk adversarial mismatch: class={stimuli_class} mode={sort_mode} "
                    f"direction={sort_direction.name} row={row}",
                    f"  expected indices:    {exp_idx.tolist()}",
                    f"  expected value bits: {_hex_row(exp_bits)}",
                    f"  result indices:      {res_indices[row].tolist()}",
                    f"  result value bits:   {_hex_row(res_value_bits[row])}",
                ]
            )
        )
