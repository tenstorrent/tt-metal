# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole dropout: the drop mask must not inherit the lane structure of the SFPU PRNG.

tt-llk#1701 item 8. The SFPU lane PRNGs are shifted views of one sequence: with the raw
draw compared straight against the probability, the mask of every SFPU row is the mask
of the row two steps earlier moved over by two lanes, bit for bit. Neighbouring elements
are then strongly correlated (about +0.3 at p=0.1), one row in seven is all-kept or
all-dropped at p=0.1, and the per-tile drop fraction spreads several times wider than
independent draws would. The kernel now salts each lane's draw and passes it through the
bijective mixer rand_tile uses (ckernel_sfpu_rand.h) before the compare.

The kernel output is deterministic for a given seed, so the statistics below are fixed
numbers per parametrization, not a sample that can flake. Each bound sits several
standard deviations of the independent-draw expectation away from it, and far below
what the unmixed kernel produced (quoted per assertion).

Blackhole only: the Wormhole dropout kernel still compares raw lane bits (same item).
"""

import math

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_UNARY_SCALAR,
    SFPU_UNARY_THRESHOLD,
    TILE_COUNT,
    generate_input_dim,
)

pytestmark = pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.BLACKHOLE,
    reason=(
        "Wormhole's dropout kernel still compares raw lane PRNG bits "
        "(tenstorrent/tt-llk#1701 item 8); Quasar has no dropout kernel"
    ),
)

INT_MAX = 2**31 - 1
TILES = 32  # 1024 SFPU rows of 32 lanes: 32768 mask bits per parametrization
TILES_PER_BLOCK = 4  # Float32 dest, SyncHalf
LANES = 32  # one SFPU row = two 16-wide rows of a face


def _mask_rows(result: torch.Tensor) -> torch.Tensor:
    """[rows, 32] drop mask in SFPU issue order.

    The result buffer is in dest order (tile, face, row, column), so consecutive groups
    of 32 elements are consecutive SFPU rows, the order the kernel draws them in.
    """
    return (result == 0.0).reshape(-1, LANES)


@pytest.mark.parametrize("probability", [0.1, 0.5], ids=lambda p: f"p{p}")
@pytest.mark.parametrize("seed", [0x12345678, 42], ids=lambda s: f"seed{s:#x}")
def test_sfpu_dropout_mask_statistics(seed, probability):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dims = [32 * TILES, 32]
    num_elements = dims[0] * dims[1]

    configuration = TestConfig(
        "sources/sfpu_dropout_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            SFPU_UNARY_SCALAR(seed),
            SFPU_UNARY_THRESHOLD(int(INT_MAX * probability)),
        ],
        runtimes=[
            TILE_COUNT(TILES),
            NUM_BLOCKS(TILES // TILES_PER_BLOCK),
            NUM_TILES_IN_BLOCK(TILES_PER_BLOCK),
        ],
        variant_stimuli=StimuliConfig(
            torch.ones(num_elements, dtype=torch.float32),
            formats.input_format,
            torch.zeros(num_elements, dtype=torch.float32),
            formats.input_format,
            formats.output_format,
            tile_count_A=TILES,
            tile_count_B=TILES,
            tile_count_res=TILES,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
    )

    result = torch.tensor(
        configuration.run().result[:num_elements], dtype=torch.float32
    )

    # scale = 1.0 on an all-ones input: anything but 0/1 is not a dropout output.
    assert torch.all(
        (result == 0.0) | (result == 1.0)
    ), "dropout output is not a 0/1 mask"

    rows = _mask_rows(result)
    p = probability
    n = rows.numel()

    # 1. Drop fraction. Unmixed kernel: 0.5251 (seed 0x12345678, p=0.5) and 0.1182 (p=0.1).
    drop_fraction = rows.float().mean().item()
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(drop_fraction - p) < 5 * sigma, (
        f"drop fraction {drop_fraction:.5f} is {abs(drop_fraction - p) / sigma:.1f} sigma "
        f"from p={p} over {n} draws"
    )

    # 2. Lane-shift structure. Unmixed kernel: row t+2 equals row t shifted by two lanes
    #    in 100% of positions. Independent draws agree with probability 1 - 2p(1-p).
    shift2_agreement = (rows[2:, 2:] == rows[:-2, :-2]).float().mean().item()
    expected_agreement = 1 - 2 * p * (1 - p)
    assert abs(shift2_agreement - expected_agreement) < 0.03, (
        f"row t+2 matches row t shifted by two lanes in {shift2_agreement:.3f} of "
        f"positions (independent draws: {expected_agreement:.3f}); the lane PRNG "
        f"structure reaches the mask"
    )

    # 3. Neighbouring elements. Unmixed kernel: mean correlation +0.31 at p=0.1.
    lanes = rows.float()
    corr = torch.corrcoef(lanes.T)
    neighbours = torch.stack(
        [corr[r * 16 + c, r * 16 + c + 1] for r in range(2) for c in range(15)]
    )
    assert neighbours.mean().abs().item() < 0.05, (
        f"mean correlation between horizontally adjacent mask elements is "
        f"{neighbours.mean().item():+.3f}"
    )

    # 4. Whole rows. Unmixed kernel: 15% of rows all-kept at p=0.1, against 0.9^32 = 3.4%.
    uniform_rows = (rows.all(dim=1) | ~rows.any(dim=1)).float().mean().item()
    expected_uniform = p**LANES + (1 - p) ** LANES
    assert uniform_rows < expected_uniform + 0.03, (
        f"{uniform_rows:.3f} of SFPU rows are entirely kept or dropped "
        f"(independent draws: {expected_uniform:.3f})"
    )

    # 5. Per-tile spread. Unmixed kernel: 3.7-6.2x the independent-draw spread.
    tile_fraction = rows.reshape(TILES, -1).float().mean(dim=1)
    iid_spread = math.sqrt(p * (1 - p) / (rows.numel() // TILES))
    assert tile_fraction.std().item() < 2 * iid_spread, (
        f"per-tile drop fraction spreads {tile_fraction.std().item() / iid_spread:.1f}x "
        f"wider than independent draws"
    )
