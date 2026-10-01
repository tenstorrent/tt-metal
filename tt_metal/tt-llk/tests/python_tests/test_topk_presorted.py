# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The tile0_sorted argument of topk_local_sort on the insertion step (sources/topk_presorted_test.cpp): the full and
the skipping build sort the same slab and are compared bit for bit, the full one also against a golden.
"""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
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
    TemplateParameter,
)
from test_topk import (
    prepare_input_tensor_for_topk,
    transform_result_tensor_to_right_form,
)

pytestmark = [skip_for_quasar]

INPUT_DIMENSIONS_SLAB = [32, 128]  # 2 value tiles and 2 index tiles per tile row
W_VALUES = 64
SLAB_TILES = 4

STIMULI_CLASSES = [
    "random",
    "all_equal",
    "few_levels",
    "neg_ties",
    "signed_zero",
    "ascending",
    "descending",
]


@dataclass
class TOPK_PRESORTED(TemplateParameter):
    tile0_sorted: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool TOPK_TILE0_SORTED = {str(self.tile0_sorted).lower()};"


def _row_values(stimuli_class, row):
    """The 64 value half of one row as bf16, deterministic per class and row."""
    generator = torch.Generator()
    generator.manual_seed(0x5E00 + 97 * row)
    if stimuli_class == "random":
        return torch.rand(W_VALUES, generator=generator).to(torch.bfloat16)
    if stimuli_class == "all_equal":
        return torch.full((W_VALUES,), 1.0, dtype=torch.bfloat16)
    if stimuli_class == "few_levels":
        levels = torch.tensor(
            [0.5, -1.0, 2.0, -0.25] * (W_VALUES // 4), dtype=torch.bfloat16
        )
    elif stimuli_class == "neg_ties":
        levels = torch.tensor(
            [-0.5, -1.0, -1.5, -2.0] * (W_VALUES // 4), dtype=torch.bfloat16
        )
    elif stimuli_class == "signed_zero":
        levels = torch.tensor(
            [0.0, -0.0, 1.0, -1.0] * (W_VALUES // 4), dtype=torch.bfloat16
        )
    elif stimuli_class == "ascending":
        return torch.arange(0, W_VALUES, dtype=torch.float32).to(torch.bfloat16)
    elif stimuli_class == "descending":
        return torch.arange(W_VALUES - 1, -1, -1, dtype=torch.float32).to(
            torch.bfloat16
        )
    else:
        raise ValueError(f"unknown stimuli class {stimuli_class}")
    perm = torch.randperm(W_VALUES, generator=generator)
    return levels[perm]


def _canonical(values):
    """The value the networks compare: -0.0 folds into +0.0 before the first compare."""
    values = values.to(torch.float32).clone()
    values[values == 0.0] = 0.0
    return values


def _stable_order(values, tie_keys, descending):
    """Positions ordered by the value in the sort direction, equal values by tie_keys ascending."""
    by_key = torch.argsort(tie_keys.to(torch.int64), stable=True)
    by_value = torch.argsort(
        values[by_key].to(torch.float32), descending=descending, stable=True
    )
    return by_key[by_value]


def _golden_second_sort(row_values, descending, tie_by_position):
    """Values and indices of the slab after the insertion step: the first sort's top 32 plus tile 1 again.
    Stable ties by the index tile, rank-stamped ties by slab position."""
    values = _canonical(row_values)
    indices = torch.arange(W_VALUES)
    order = _stable_order(
        values, indices, descending
    )  # first sort: the positions are the indices
    top_values = values[order][:32]
    top_indices = indices[order][:32]
    values2 = torch.cat([top_values, values[32:]])
    indices2 = torch.cat([top_indices, indices[32:]])
    order2 = _stable_order(
        values2, torch.arange(W_VALUES) if tie_by_position else indices2, descending
    )
    return values2[order2], indices2[order2]


def _config(
    formats,
    sort_direction,
    sort_mode,
    src_A,
    tile_cnt_A,
    src_B,
    tile_cnt_B,
    tile0_sorted,
):
    return TestConfig(
        test_name="sources/topk_presorted_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=64,  # end phase 5, four result tiles
                topk_matrix_width=INPUT_DIMENSIONS_SLAB[1],
                topk_sort_direction=sort_direction,
                topk_stable_sort=sort_mode == "stable",
                topk_fused_stable=False,
                topk_rank_stamped=sort_mode == "rank_stamped",
            ),
            TOPK_PRESORTED(tile0_sorted=tile0_sorted),
        ],
        runtimes=[
            INPUT_DIMENSIONS(
                INPUT_DIMENSIONS_SLAB[0] // 32, INPUT_DIMENSIONS_SLAB[1] // 32
            ),
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
            tile_count_res=SLAB_TILES,
        ),
        dest_acc=(
            DestAccumulation.Yes if sort_mode == "rank_stamped" else DestAccumulation.No
        ),
        unpack_to_dest=False,
    )


def _run(configuration, formats):
    res = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    res = transform_result_tensor_to_right_form(res, formats, 64, INPUT_DIMENSIONS_SLAB)
    untilizer = get_golden_generator(UntilizeGolden)
    rows = untilizer(
        res, formats.output_format, [INPUT_DIMENSIONS_SLAB[0], 2 * W_VALUES]
    ).reshape(INPUT_DIMENSIONS_SLAB[0], 2 * W_VALUES)
    values = rows[:, :W_VALUES].contiguous()
    indices = rows[:, W_VALUES:].contiguous().view(torch.uint16).to(torch.int64)
    return values, indices


@parametrize(
    sort_direction=[TopKSortDirection.Descending, TopKSortDirection.Ascending],
    sort_mode=["unstable", "stable", "rank_stamped"],
    stimuli_class=STIMULI_CLASSES,
)
def test_topk_presorted(
    sort_direction: TopKSortDirection, sort_mode: str, stimuli_class: str
):
    formats: InputOutputFormat = input_output_formats([DataFormat.Float16_b])[0]
    descending = sort_direction == TopKSortDirection.Descending
    torch.manual_seed(0)

    num_rows, num_cols = INPUT_DIMENSIONS_SLAB
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS_SLAB,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS_SLAB,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
        spec_B=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    row_values = torch.stack(
        [_row_values(stimuli_class, row) for row in range(num_rows)]
    )
    src_A = src_A.clone().view(num_rows, num_cols)
    src_A[:, :W_VALUES] = row_values.to(src_A.dtype)
    src_A = src_A.flatten()
    src_A = prepare_input_tensor_for_topk(src_A, formats, INPUT_DIMENSIONS_SLAB)

    # Both builds are prepared before either runs: the compile-producer pass ends the test at the first run().
    full = _config(
        formats, sort_direction, sort_mode, src_A, tile_cnt_A, src_B, tile_cnt_B, False
    )
    skip = _config(
        formats, sort_direction, sort_mode, src_A, tile_cnt_A, src_B, tile_cnt_B, True
    )
    full.prepare()
    skip.prepare()
    full_values, full_indices = _run(full, formats)
    skip_values, skip_indices = _run(skip, formats)

    # The two builds against each other, bit for bit.
    assert torch.equal(
        full_values.view(torch.int16), skip_values.view(torch.int16)
    ), "the values of the two sorts differ"
    if sort_mode in ("stable", "rank_stamped"):
        assert torch.equal(
            full_indices, skip_indices
        ), "the indices of the two stable sorts differ"
    else:
        for row in range(num_rows):
            distinct = torch.unique(_canonical(row_values[row])).numel() == W_VALUES
            if distinct:
                assert torch.equal(
                    full_indices[row], skip_indices[row]
                ), f"row {row} without equal values: the indices differ"

    # The full sort against its golden.
    for row in range(num_rows):
        expected_values, expected_indices = _golden_second_sort(
            row_values[row], descending, sort_mode == "rank_stamped"
        )
        got_values = full_values[row].to(torch.float32)
        assert torch.equal(
            got_values, expected_values
        ), f"row {row}: values differ from the golden"
        if sort_mode in ("stable", "rank_stamped"):
            assert torch.equal(
                full_indices[row], expected_indices
            ), f"row {row}: indices differ from the stable golden"
        else:
            source = _canonical(row_values[row])
            assert torch.equal(
                source[full_indices[row]], got_values
            ), f"row {row}: an index does not name its value"
