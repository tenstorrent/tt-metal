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

import os
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
from helpers.llk_params import (
    DestAccumulation,
    PerfRunType,
    TopKSortDirection,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
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


def make_unique_value_input(src_A, input_dimensions=[32, 128], order="ascending"):
    """Give each row unique, exactly representable values for an exact index gate."""
    src_A = src_A.clone()
    num_rows_tensor, num_cols_tensor = input_dimensions
    values_per_row = num_cols_tensor // NUM_STAGES
    unique_values = torch.arange(values_per_row, dtype=torch.float32).to(src_A.dtype)
    for row in range(num_rows_tensor):
        values_start_idx = row * num_cols_tensor
        values = unique_values
        if order == "descending":
            values = values.flip(0)
        elif order == "permuted":
            generator = torch.Generator().manual_seed(42 + row)
            values = (values - values_per_row // 2)[
                torch.randperm(values_per_row, generator=generator)
            ]
        src_A[values_start_idx : values_start_idx + values_per_row] = values
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
    K=[32],  # TODO: Add more K values (like 16, 64).
    sort_direction=[TopKSortDirection.Descending, TopKSortDirection.Ascending],
    stable_sort=[False, True],
    implementation=[0, 1],
)
def test_topk_sfpu(
    formats: InputOutputFormat,
    input_dimensions: list,
    K: int,
    sort_direction: TopKSortDirection,
    stable_sort: bool,
    implementation: int,
    exact_order=None,
):

    if stable_sort:
        pytest.skip(
            "Stable sort is currently not broken in LLK API."
        )  # TODO: Check tenstorrent/tt-metal#33492 and remove this once fixed.

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    if exact_order is not None or os.getenv("TOPK_EXACT_UNIQUE") == "1":
        src_A = make_unique_value_input(
            src_A, input_dimensions, exact_order or "ascending"
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
                topk_impl=implementation,
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

    if exact_order is not None:
        # Unique exactly representable values make values/indices unambiguous.
        assert torch.equal(res_tensor, golden_tensor), "exact TopK value/index mismatch"

    # Widths above 128 loop results back through L1; topk_test.cpp orders that
    # write-back with PACK_DONE (previously the source of issue #1344).
    if not _RECORD_TEST_ORDER:
        assert validate_topk_indices(
            res_tensor, golden_tensor, src_A, formats, input_dimensions, K, stable_sort
        )

    # Get value tiles from result and golden tensors
    res_values = get_value_tiles_from_topk_tensor(res_tensor, K, input_dimensions)
    golden_values = get_value_tiles_from_topk_tensor(golden_tensor, K, input_dimensions)

    # Validate topk values
    assert passed_test(
        golden_values, res_values, formats.output_format, print_errors=True
    )


@pytest.mark.parametrize(
    "implementation,label",
    [(0, "handwritten"), (1, "typed_multiresult"), (2, "threaded_merge")],
)
def test_topk_device_profile(perf_report, implementation: int, label: str):
    """Profile one 32x128 TopK SFPU body, excluding datacopy and handshakes."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_dimensions = [32, 128]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = make_unique_value_input(src_A, input_dimensions)
    src_A = prepare_input_tensor_for_topk(src_A, formats, input_dimensions)

    configuration = PerfConfig(
        "sources/topk_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=32,
                topk_matrix_width=128,
                topk_sort_direction=TopKSortDirection.Descending,
                topk_stable_sort=False,
                topk_impl=implementation,
            ),
        ],
        runtimes=[INPUT_DIMENSIONS(1, 4), TILE_COUNT(tile_cnt_A)],
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
    configuration.run(perf_report, run_count=5)
    rows = perf_report.frame()
    rows = rows[rows["marker"] == "TOPK_BODY"]
    assert len(rows) >= 1, rows.to_string(index=False)
    cycles = float(rows.iloc[-1]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(f"TOPK_DEVICE_PROFILE impl={label} body_cycles={cycles:.2f}")


@pytest.mark.parametrize(
    "implementation",
    [0, 2, 3],
    ids=["handwritten", "threaded_merge", "threaded_merge_stress"],
)
@pytest.mark.parametrize("rows", [32, 64])
@pytest.mark.parametrize("order", ["ascending", "descending", "permuted"])
@pytest.mark.parametrize(
    "direction", [TopKSortDirection.Descending, TopKSortDirection.Ascending]
)
@pytest.mark.parametrize("data_format", [DataFormat.Float16_b, DataFormat.Float16])
def test_topk_threaded_merge_exact(
    implementation, rows, order, direction, data_format, monkeypatch
):
    """Production TopK pipeline, replacing only the four-register merge region."""
    monkeypatch.setattr(TestConfig, "BIT_EXACT_RUNS", max(2, TestConfig.BIT_EXACT_RUNS))
    test_topk_sfpu(
        InputOutputFormat(data_format, data_format),
        [rows, 128],
        32,
        direction,
        False,
        implementation,
        exact_order=order,
    )


# ---------------------------------------------------------------------------
# Explicit-state merge (TOPK_IMPL=2) against the handwritten merge (TOPK_IMPL=0)
# outside the 72-case exact gate: ties, stable sorting, special values, wider
# rows (merge m_iter > 0), K=64 and 32-bit DEST. Both arms replace only the
# merge region and issue the same instruction words, so the primary gate is
# bitwise equality of the full packed result on identical stimuli. Golden
# checks are added only where the golden's semantics are well defined.
# ---------------------------------------------------------------------------

_SPECIAL_BITS = {
    DataFormat.Float16_b: {
        "pos_inf": 0x7F80,
        "neg_inf": 0xFF80,
        "pos_zero": 0x0000,
        "neg_zero": 0x8000,
        "pos_subnormal": 0x0001,
        "neg_subnormal": 0x8001,
        "max_normal": 0x7F7F,
        "neg_max_normal": 0xFF7F,
        "nan": 0x7FC0,
    },
    DataFormat.Float16: {
        "pos_inf": 0x7C00,
        "neg_inf": 0xFC00,
        "pos_zero": 0x0000,
        "neg_zero": 0x8000,
        "pos_subnormal": 0x0001,
        "neg_subnormal": 0x8001,
        "max_normal": 0x7BFF,
        "neg_max_normal": 0xFBFF,
        "nan": 0x7E00,
    },
}


def _bits_to_values(bits, dtype):
    return torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(dtype)


def _fill_rows(src_A, input_dimensions, row_values):
    """row_values(row) -> 1D tensor of values_per_row values in src_A.dtype."""
    src_A = src_A.clone()
    rows, cols = input_dimensions
    per_row = cols // NUM_STAGES
    for row in range(rows):
        vals = row_values(row)
        assert vals.numel() == per_row and vals.dtype == src_A.dtype
        src_A[row * cols : row * cols + per_row] = vals
    return src_A


def _tie_rows(data_format, input_dimensions, pattern):
    dtype = format_dict[data_format]
    per_row = input_dimensions[1] // NUM_STAGES

    def values(row):
        g = torch.Generator().manual_seed(1000 + row)
        if pattern == "all_equal":
            return torch.full((per_row,), 1.5, dtype=torch.float32).to(dtype)
        if pattern == "few_levels":
            # Four levels: every top-K boundary falls inside a tie group.
            return (
                torch.randint(0, 4, (per_row,), generator=g).to(torch.float32).to(dtype)
            )
        if pattern == "pairs":
            # Each value appears exactly twice, positions shuffled.
            base = torch.arange(per_row // 2, dtype=torch.float32).repeat(2)
            return base[torch.randperm(per_row, generator=g)].to(dtype)
        raise ValueError(pattern)

    return values


def _special_rows(data_format, input_dimensions, pattern):
    dtype = format_dict[data_format]
    bits = _SPECIAL_BITS[data_format]
    per_row = input_dimensions[1] // NUM_STAGES

    def values(row):
        g = torch.Generator().manual_seed(2000 + row)
        base = (torch.arange(per_row, dtype=torch.float32) - per_row // 2)[
            torch.randperm(per_row, generator=g)
        ].to(dtype)
        if pattern == "inf_zero":
            names = [
                "pos_inf",
                "neg_inf",
                "pos_zero",
                "neg_zero",
                "max_normal",
                "neg_max_normal",
            ]
        elif pattern == "subnormal":
            names = ["pos_subnormal", "neg_subnormal", "pos_zero", "neg_zero"]
        elif pattern == "nan":
            names = ["nan", "pos_inf", "neg_inf"]
        else:
            raise ValueError(pattern)
        specials = _bits_to_values([bits[n] for n in names], dtype)
        positions = torch.randperm(per_row, generator=g)[: len(names)]
        base[positions] = specials
        return base

    return values


def _run_topk_raw(
    formats,
    input_dimensions,
    K,
    sort_direction,
    stable_sort,
    implementation,
    src_A,
    dest_acc,
):
    """Run one TopK variant and return (result tensor in golden layout, prepared input)."""
    _, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    prepared = prepare_input_tensor_for_topk(src_A, formats, input_dimensions)
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
                topk_impl=implementation,
            ),
        ],
        runtimes=[
            INPUT_DIMENSIONS(input_dimensions[0] // 32, input_dimensions[1] // 32),
            TILE_COUNT(tile_cnt_A),
        ],
        variant_stimuli=StimuliConfig(
            prepared,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )
    res = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    return (
        transform_result_tensor_to_right_form(res, formats, K, input_dimensions),
        prepared,
    )


def _bits(t):
    return t.contiguous().view(torch.int16)


def _first_bit_mismatch(a, b):
    ab, bb = _bits(a).reshape(-1), _bits(b).reshape(-1)
    diff = (ab != bb).nonzero().flatten()
    if diff.numel() == 0:
        return None
    i = int(diff[0])
    return f"{diff.numel()} differing elements; first at {i}: {int(ab[i]) & 0xFFFF:#06x} vs {int(bb[i]) & 0xFFFF:#06x}"


def _golden_value_check(
    res, golden, formats, input_dimensions, K, allow_signed_zero=False
):
    """Values must match the golden bitwise (optionally treating +0/-0 alike)."""
    rv = get_value_tiles_from_topk_tensor(res, K, input_dimensions)
    gv = get_value_tiles_from_topk_tensor(golden, K, input_dimensions)
    rb, gb = _bits(rv).reshape(-1), _bits(gv).reshape(-1)
    if allow_signed_zero:
        zero = torch.tensor(0x7FFF, dtype=torch.int16)
        rz, gz = (rb & zero) == 0, (gb & zero) == 0
        rb = torch.where(rz, torch.zeros_like(rb), rb)
        gb = torch.where(gz, torch.zeros_like(gb), gb)
    return _first_bit_mismatch(rb.view(rv.dtype), gb.view(gv.dtype))


def _assert_fp16_inf_order_defect(res, golden, formats, input_dimensions, K, direction):
    """Refuse to XFAIL anything except the measured inf/max-normal swap."""
    rv = get_value_tiles_from_topk_tensor(res, K, input_dimensions)
    gv = get_value_tiles_from_topk_tensor(golden, K, input_dimensions)
    rb, gb = _bits(rv).reshape(-1), _bits(gv).reshape(-1)
    # Signed zero is not part of this defect and is already licensed by the
    # caller's golden-value comparison.
    zero_mask = torch.tensor(0x7FFF, dtype=torch.int16)
    rb = torch.where((rb & zero_mask) == 0, torch.zeros_like(rb), rb)
    gb = torch.where((gb & zero_mask) == 0, torch.zeros_like(gb), gb)
    diff = (rb != gb).nonzero().flatten()
    assert (
        diff.numel() == 2 * input_dimensions[0]
    ), f"FP16 inf XFAIL shape changed: expected two swapped values per row, got {diff.numel()}"
    expected = (
        {0x7C00, 0x7BFF}
        if direction == TopKSortDirection.Descending
        else {0xFC00, 0xFBFF}
    )
    for observed, label in ((rb[diff], "device"), (gb[diff], "golden")):
        values = [int(v) & 0xFFFF for v in observed]
        assert set(values) == expected, (
            f"FP16 inf XFAIL absorbed an unrelated {label} mismatch: "
            f"expected only {sorted(expected)}, got {sorted(set(values))}"
        )
        for value in expected:
            assert (
                values.count(value) == input_dimensions[0]
            ), f"FP16 inf XFAIL {label} multiplicity changed for {value:#06x}"


def _explicit_vs_hand(
    data_format,
    input_dimensions,
    K,
    direction,
    stable_sort,
    src_values,
    dest_acc=DestAccumulation.No,
):
    formats = InputOutputFormat(data_format, data_format)
    src_A, _, _, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = _fill_rows(src_A, input_dimensions, src_values)
    golden = get_golden_generator(TopKGolden)(
        src_A, data_format, K, direction, input_dimensions=input_dimensions
    )
    hand, prepared = _run_topk_raw(
        formats, input_dimensions, K, direction, stable_sort, 0, src_A, dest_acc
    )
    explicit, _ = _run_topk_raw(
        formats, input_dimensions, K, direction, stable_sort, 2, src_A, dest_acc
    )
    return formats, hand, explicit, golden, prepared


_TIE_DIMS = [[32, 128], [32, 256], [32, 512]]


@pytest.mark.parametrize("data_format", [DataFormat.Float16_b, DataFormat.Float16])
@pytest.mark.parametrize(
    "direction", [TopKSortDirection.Descending, TopKSortDirection.Ascending]
)
@pytest.mark.parametrize("stable_sort", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("pattern", ["all_equal", "few_levels", "pairs"])
@pytest.mark.parametrize("dims", _TIE_DIMS, ids=lambda d: f"{d[0]}x{d[1]}")
def test_topk_explicit_ties(
    data_format, direction, stable_sort, pattern, dims, record_property
):
    formats, hand, explicit, golden, prepared = _explicit_vs_hand(
        data_format,
        dims,
        32,
        direction,
        stable_sort,
        _tie_rows(data_format, dims, pattern),
    )
    mismatch = _first_bit_mismatch(explicit, hand)
    assert mismatch is None, f"explicit != handwritten: {mismatch}"
    # Tie order may legitimately differ from the golden's stable order at this
    # checkpoint: full stable TopK support was reverted.  Still gate the value
    # multiset, index/value association, and explicit-vs-hand equivalence, and
    # record exact stable ordering so a future implementation cannot hide it.
    assert _golden_value_check(explicit, golden, formats, dims, 32) is None
    assert validate_topk_indices(
        explicit, golden, prepared, formats, dims, 32, stable_sort=False
    )
    stable_exact = torch.equal(_bits(explicit), _bits(golden))
    record_property("stable_index_order_matches_golden", stable_exact)
    print(
        f"TOPK_TIES pattern={pattern} dims={dims} stable={stable_sort} golden_index_order_exact={stable_exact}"
    )


@pytest.mark.parametrize("data_format", [DataFormat.Float16_b, DataFormat.Float16])
@pytest.mark.parametrize(
    "direction", [TopKSortDirection.Descending, TopKSortDirection.Ascending]
)
@pytest.mark.parametrize("pattern", ["inf_zero", "subnormal", "nan"])
def test_topk_explicit_special_values(data_format, direction, pattern, record_property):
    dims = [32, 128]
    formats, hand, explicit, golden, prepared = _explicit_vs_hand(
        data_format,
        dims,
        32,
        direction,
        False,
        _special_rows(data_format, dims, pattern),
    )
    mismatch = _first_bit_mismatch(explicit, hand)
    assert mismatch is None, f"explicit != handwritten: {mismatch}"
    # NaN has no golden ordering; subnormal handling is a hardware load-format
    # property shared by both arms. Record the golden comparison for both.
    golden_mismatch = _golden_value_check(
        explicit, golden, formats, dims, 32, allow_signed_zero=True
    )
    record_property("golden_value_mismatch", str(golden_mismatch))
    print(
        f"TOPK_SPECIAL pattern={pattern} fmt={data_format.name} dir={direction.name} golden_value_mismatch={golden_mismatch}"
    )
    if pattern == "inf_zero":
        if data_format == DataFormat.Float16 and golden_mismatch is not None:
            # Measured on Blackhole for both arms: selection and indices are
            # correct and +/-inf keep their bits, but the shared SFPSWAP path
            # ranks +/-max-normal ahead of +/-inf for FP16 (BF16 is correct).
            _assert_fp16_inf_order_defect(
                explicit, golden, formats, dims, 32, direction
            )
            pytest.xfail(
                f"FP16 inf ranked below max-normal (shared by handwritten): {golden_mismatch}"
            )
        assert golden_mismatch is None, golden_mismatch


@pytest.mark.parametrize("data_format", [DataFormat.Float16_b, DataFormat.Float16])
@pytest.mark.parametrize(
    "direction", [TopKSortDirection.Descending, TopKSortDirection.Ascending]
)
@pytest.mark.parametrize("stable_sort", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("K", [32, 64])
@pytest.mark.parametrize(
    "dims",
    [[32, 256], [32, 512], [64, 256], [32, 1024]],
    ids=lambda d: f"{d[0]}x{d[1]}",
)
def test_topk_explicit_widths(data_format, direction, stable_sort, K, dims):
    if K * NUM_STAGES > dims[1] // NUM_STAGES:
        pytest.skip("K wider than half the value columns")
    per_row = dims[1] // NUM_STAGES
    dtype = format_dict[data_format]

    def values(row):
        # Unique, exactly representable in BF16/FP16 (|v| <= 256).
        g = torch.Generator().manual_seed(3000 + row)
        return (torch.arange(per_row, dtype=torch.float32) - per_row // 2)[
            torch.randperm(per_row, generator=g)
        ].to(dtype)

    formats, hand, explicit, golden, _ = _explicit_vs_hand(
        data_format, dims, K, direction, stable_sort, values
    )
    mismatch = _first_bit_mismatch(explicit, hand)
    assert mismatch is None, f"explicit != handwritten: {mismatch}"
    golden_mismatch = _first_bit_mismatch(explicit, golden)
    if K > 32:
        # topk_test.cpp packs one value tile and one index tile per row in the
        # final iteration and the merge clamps k to 32, so K>32 has no golden
        # layout here. The explicit/handwritten bitwise gate above still holds.
        pytest.xfail(
            f"K={K} output layout unsupported by topk_test.cpp; golden={golden_mismatch}"
        )
    print(
        f"TOPK_WIDTH dims={dims} K={K} stable={stable_sort} fmt={data_format.name} dir={direction.name} golden={golden_mismatch}"
    )
    assert golden_mismatch is None, f"explicit != golden: {golden_mismatch}"


@pytest.mark.parametrize(
    "direction", [TopKSortDirection.Descending, TopKSortDirection.Ascending]
)
@pytest.mark.parametrize("stable_sort", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("dims", [[32, 128], [32, 256]], ids=lambda d: f"{d[0]}x{d[1]}")
def test_topk_explicit_fp32_dest(direction, stable_sort, dims):
    data_format = DataFormat.Float16_b
    per_row = dims[1] // NUM_STAGES
    dtype = format_dict[data_format]

    def values(row):
        g = torch.Generator().manual_seed(4000 + row)
        return (torch.arange(per_row, dtype=torch.float32) - per_row // 2)[
            torch.randperm(per_row, generator=g)
        ].to(dtype)

    formats, hand, explicit, golden, _ = _explicit_vs_hand(
        data_format,
        dims,
        32,
        direction,
        stable_sort,
        values,
        dest_acc=DestAccumulation.Yes,
    )
    mismatch = _first_bit_mismatch(explicit, hand)
    assert mismatch is None, f"explicit != handwritten: {mismatch}"
    golden_mismatch = _first_bit_mismatch(explicit, golden)
    print(
        f"TOPK_FP32DEST dims={dims} stable={stable_sort} dir={direction.name} golden={golden_mismatch}"
    )
    assert golden_mismatch is None, f"explicit != golden: {golden_mismatch}"
