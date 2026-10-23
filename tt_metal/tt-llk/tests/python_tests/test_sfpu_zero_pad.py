# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Zero-pad SFPU test. Covers experimental/ckernel_sfpu_zero_pad.h.

The kernel skips VALID_ROWS SFPU rows without writing, then stores 0.0 into rows
[VALID_ROWS, TOTAL_ROWS). Rows are counted in Dest face order, so VALID_ROWS=8
zeroes the bottom half of the tile plus its top-right quadrant, not the last 24
tile rows.

VALID_ROWS and TOTAL_ROWS have to be even.
"""

import torch
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT, ZERO_PAD_ROWS

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

SFPU_ROW_ELEMENTS = 32
SFPU_ROWS_PER_TILE = ELEMENTS_PER_TILE // SFPU_ROW_ELEMENTS


def _valid_dest_acc(formats):
    if formats.input.is_32_bit():
        return [DestAccumulation.Yes]
    return get_valid_dest_accumulation_modes(formats)


def _zero_pad_golden(tile, valid_rows: int, total_rows: int):
    golden = tile.to(torch.float32).clone()
    golden[SFPU_ROW_ELEMENTS * valid_rows : SFPU_ROW_ELEMENTS * total_rows] = 0.0
    return golden


# (valid_rows, total_rows).
ROW_RANGES = [
    (0, 32),  # zero the entire tile
    (2, 32),
    (8, 32),  # face boundary
    (14, 32),
    (16, 32),  # half-tile boundary: the bottom half of the tile
    (24, 32),
    (30, 32),  # a minimal trailing range
    (32, 32),  # empty range: the kernel must write nothing at all
    (0, 2),  # a minimal leading range
    (2, 4),
    # total_rows < 32, so the untouched-tail assertion has something to check:
    # faces 2 and 3 must survive untouched.
    (0, 16),
    (8, 16),
    (14, 16),
    (16, 16),  # empty range, and faces 2-3 must survive
]


def _build_input_tile(torch_format) -> torch.Tensor:
    """Flat tilized tile whose every datum is non-zero and tags its SFPU row."""
    rows = torch.arange(ELEMENTS_PER_TILE) // SFPU_ROW_ELEMENTS + 1
    return rows.to(torch.float32).to(torch_format)


def _run(
    formats, dest_acc, valid_rows, total_rows
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one variant, returning (result, golden) as flat tilized float32 tiles."""
    torch_format = format_dict[formats.input_format]

    src_A = _build_input_tile(torch_format)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    golden = _zero_pad_golden(src_A, valid_rows, total_rows)

    configuration = TestConfig(
        "sources/sfpu_zero_pad_test.cpp",
        formats,
        templates=[
            ZERO_PAD_ROWS(valid_rows=valid_rows, total_rows=total_rows),
        ],
        runtimes=[
            TILE_COUNT(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    res_from_L1 = configuration.run().result
    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    return res.to(torch.float32), golden


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=lambda formats: _valid_dest_acc(formats),
    row_range=ROW_RANGES,
)
def test_sfpu_zero_pad(formats, dest_acc, row_range):
    valid_rows, total_rows = row_range
    res, golden = _run(formats, dest_acc, valid_rows, total_rows)

    padded = res[SFPU_ROW_ELEMENTS * valid_rows : SFPU_ROW_ELEMENTS * total_rows]
    assert bool((padded == 0.0).all()), (
        f"SFPU rows [{valid_rows}, {total_rows}) must come back exactly 0.0, "
        f"got non-zero at flat offsets "
        f"{(padded != 0.0).nonzero().flatten().tolist()[:16]}"
    )

    head = res[: SFPU_ROW_ELEMENTS * valid_rows]
    tail = res[SFPU_ROW_ELEMENTS * total_rows :]
    assert bool((head == golden[: SFPU_ROW_ELEMENTS * valid_rows]).all()), (
        f"SFPU rows [0, {valid_rows}) are skipped without a store and must be "
        f"unchanged"
    )
    assert bool((tail == golden[SFPU_ROW_ELEMENTS * total_rows :]).all()), (
        f"SFPU rows [{total_rows}, {SFPU_ROWS_PER_TILE}) are never reached and "
        f"must be unchanged"
    )
