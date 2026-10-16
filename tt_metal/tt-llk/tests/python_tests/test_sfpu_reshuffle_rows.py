# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""reshuffle_rows scatter-add output[idx[i]] += input[i] over a tile (255 skips a row)."""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import RESHUFFLE_ROWS_FORM, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block

FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
SKIP = 255
TILE_DIMS = [TILE_DIM, TILE_DIM]
# Small integers keep every partial sum exact in bf16.
MAX_ABS_VALUE = 4


def _index_patterns():
    generator = torch.Generator().manual_seed(7)
    random_permutation = torch.randperm(TILE_DIM, generator=generator).tolist()
    mixed = [SKIP if r % 5 == 0 else (r * 7) % 11 for r in range(TILE_DIM)]
    # Indices of 32 and above skip their rows like 255.
    out_of_range = [31, 32, 33, 47] + [
        32 + r // 2 if r % 2 else r for r in range(4, TILE_DIM)
    ]
    return {
        "identity": list(range(TILE_DIM)),
        "reversed": list(range(TILE_DIM - 1, -1, -1)),
        "random_permutation": random_permutation,
        "every_second_row_skipped": [SKIP if r % 2 else r for r in range(TILE_DIM)],
        "all_rows_skipped": [SKIP] * TILE_DIM,
        "all_rows_into_row_5": [5] * TILE_DIM,
        "mixed_skipped_and_shared": mixed,
        "out_of_range_indices": out_of_range,
    }


INDEX_PATTERNS = _index_patterns()


def _index_tile(indices, torch_format):
    """A bf16 tile whose first 32 bytes in L1 are the index bytes, two per 16-bit element, little endian."""
    words = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.int16)
    for k in range(TILE_DIM // 2):
        low, high = indices[2 * k], indices[2 * k + 1]
        value = low | (high << 8)
        if value >= 0x8000:
            value -= 0x10000
        words[k] = value
    return words.view(torch.bfloat16).to(torch_format)


def _golden(input_rows, output_rows, indices):
    golden = output_rows.to(torch.float32).clone()
    for row, destination in enumerate(indices):
        if destination < TILE_DIM:
            golden[destination] += input_rows[row].to(torch.float32)
    return golden


def _run(formats, dest_acc, indices, replay):
    generator = torch.Generator().manual_seed(3)
    torch_format = format_dict[formats.input_format]
    shape = (TILE_DIM, TILE_DIM)
    input_rows = torch.randint(
        -MAX_ABS_VALUE, MAX_ABS_VALUE + 1, shape, generator=generator
    ).to(torch.float32)
    output_rows = torch.randint(
        -MAX_ABS_VALUE, MAX_ABS_VALUE + 1, shape, generator=generator
    ).to(torch.float32)
    golden_rows = _golden(input_rows, output_rows, indices)

    src_A = tilize_block(
        input_rows.to(torch_format), TILE_DIMS, stimuli_format=formats.input_format
    ).flatten()
    index_tile = _index_tile(indices, torch_format).flatten()
    accumulator = tilize_block(
        output_rows.to(torch_format), TILE_DIMS, stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.cat([index_tile, accumulator])

    configuration = TestConfig(
        "sources/sfpu_reshuffle_rows_test.cpp",
        formats,
        templates=[RESHUFFLE_ROWS_FORM(replay)],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=2,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    res_rows = (
        untilize_block(res, formats.output_format, TILE_DIMS)
        .reshape(TILE_DIM, TILE_DIM)
        .to(torch.float32)
    )
    return res_rows, golden_rows


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    pattern=sorted(INDEX_PATTERNS),
    replay=[False, True],
)
def test_sfpu_reshuffle_rows(formats, dest_acc, pattern, replay):
    if replay and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("the replayed form is Blackhole only")
    indices = INDEX_PATTERNS[pattern]
    res_rows, golden_rows = _run(formats, dest_acc, indices, replay)
    bad_rows = (res_rows != golden_rows).any(dim=1).nonzero().flatten().tolist()
    assert (
        not bad_rows
    ), f"reshuffle_rows ({'replayed' if replay else 'direct'}) with the {pattern} index pattern: output rows {bad_rows} differ from the golden"
