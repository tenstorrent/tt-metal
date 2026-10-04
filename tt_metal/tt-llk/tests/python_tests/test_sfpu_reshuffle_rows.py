# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
reshuffle_rows SFPU test: the scatter-add output[idx[i]] += input[i] over the 32 rows of a tile,
with the destination row indices in the first 32 bytes of buffer_B[0] (255 skips the row). Small
integers keep every partial sum exact in bf16; the patterns cover permutations, skipped and shared rows.
"""

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block

FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
SKIP = 255
TILE_DIMS = [TILE_DIM, TILE_DIM]


def _index_patterns():
    generator = torch.Generator().manual_seed(7)
    random_permutation = torch.randperm(TILE_DIM, generator=generator).tolist()
    mixed = [SKIP if r % 5 == 0 else (r * 7) % 11 for r in range(TILE_DIM)]
    return {
        "identity": list(range(TILE_DIM)),
        "reversed": list(range(TILE_DIM - 1, -1, -1)),
        "random_permutation": random_permutation,
        "every_second_row_skipped": [SKIP if r % 2 else r for r in range(TILE_DIM)],
        "all_rows_skipped": [SKIP] * TILE_DIM,
        "all_rows_into_row_5": [5] * TILE_DIM,
        "mixed_skipped_and_shared": mixed,
    }


INDEX_PATTERNS = _index_patterns()


def _index_tile(indices, torch_format):
    """A bf16 tile whose first 32 bytes in L1 are the 32 index bytes: two bytes per 16-bit
    element, little endian (row 2k in the low half, row 2k + 1 in the high half)."""
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


def _run(formats, dest_acc, indices):
    generator = torch.Generator().manual_seed(3)
    torch_format = format_dict[formats.input_format]
    # Integers in [-4, 4]: 32 accumulations stay below 2^8, so every partial sum is exact in bf16.
    input_rows = torch.randint(-4, 5, (TILE_DIM, TILE_DIM), generator=generator).to(
        torch.float32
    )
    output_rows = torch.randint(-4, 5, (TILE_DIM, TILE_DIM), generator=generator).to(
        torch.float32
    )
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
        templates=[],
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
)
def test_sfpu_reshuffle_rows(formats, dest_acc, pattern):
    indices = INDEX_PATTERNS[pattern]
    res_rows, golden_rows = _run(formats, dest_acc, indices)
    bad_rows = (res_rows != golden_rows).any(dim=1).nonzero().flatten().tolist()
    assert (
        not bad_rows
    ), f"reshuffle_rows with the {pattern} index pattern: output rows {bad_rows} differ from the golden"
