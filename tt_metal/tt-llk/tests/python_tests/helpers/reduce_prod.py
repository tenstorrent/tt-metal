# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stimuli and result bounds for the scalar PROD SFPU reduce, shared by the WH/BH and Quasar tests."""

import torch

from .format_config import InputOutputFormat
from .golden_generators import ELEMENTS_PER_TILE, TILE_DIM, truncate_to_dest_width
from .llk_params import DestAccumulation, format_dict

PROD_DIMENSIONS = [[TILE_DIM, TILE_DIM], [2 * TILE_DIM, 2 * TILE_DIM]]
PROD_STIMULI = ["pow2", "near_one"]

PROD_STIMULUS_SEED = 48794
# A pow2 tile t holds this many 0.5s and this many plus t + 1 2s; every other value is 1.
POW2_HALVES_PER_TILE = 64
# near_one magnitude range: wide enough that the fp32 multiply chain rounds, narrow enough that a
# tile's product stays inside fp16's range.
NEAR_ONE_LOW = 0.9
NEAR_ONE_HIGH = 1.1
# Relative error the ~1023 fp32 multiplies may build up before the store. Covers 1023 * 2**-23,
# the worst case even if every multiply truncated.
PROD_FP32_CHAIN_RTOL = 3e-4


def generate_prod_stimuli(stimulus: str, tile_cnt: int) -> torch.Tensor:
    """Per-tile PROD stimuli, laid out tile by tile (the product does not depend on the order).

    ``pow2``: +-1, +-2 and +-0.5 only, so every partial product is an exact power of two and a
    lost sign or a dropped or duplicated +-2 or +-0.5 shows up as a bit mismatch.
    ``near_one``: random signs on magnitudes in (NEAR_ONE_LOW, NEAR_ONE_HIGH), so the fp32
    multiply chain rounds.
    """
    gen = torch.Generator().manual_seed(PROD_STIMULUS_SEED)
    tiles = []
    for t in range(tile_cnt):
        if stimulus == "pow2":
            twos = POW2_HALVES_PER_TILE + t + 1
            magnitudes = torch.ones(ELEMENTS_PER_TILE)
            magnitudes[:twos] = 2.0
            magnitudes[twos : twos + POW2_HALVES_PER_TILE] = 0.5
            magnitudes = magnitudes[torch.randperm(ELEMENTS_PER_TILE, generator=gen)]
        else:
            magnitudes = torch.empty(ELEMENTS_PER_TILE).uniform_(
                NEAR_ONE_LOW, NEAR_ONE_HIGH, generator=gen
            )
        signs = torch.where(
            torch.rand(ELEMENTS_PER_TILE, generator=gen) < 0.5, -1.0, 1.0
        )
        tiles.append(magnitudes * signs)
    return torch.cat(tiles)


def prod_result_bounds(
    src: torch.Tensor,
    tile_cnt: int,
    stimulus: str,
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest and highest value each tile's element [0][0] may hold.

    The exact product, widened by PROD_FP32_CHAIN_RTOL for near_one (pow2 products are exact),
    then narrowed as the device stores it: SFPSTORE into a 16-bit Dest truncates toward zero,
    while a 32-bit Dest keeps fp32 and the packer rounds to the output format. Both narrowings
    are monotonic, so narrowing the two ends bounds every value the fp32 chain can produce.
    """
    exact = src.to(torch.float64).view(tile_cnt, ELEMENTS_PER_TILE).prod(dim=1)
    rtol = 0.0 if stimulus == "pow2" else PROD_FP32_CHAIN_RTOL
    ends = torch.stack((exact * (1 - rtol), exact * (1 + rtol))).to(torch.float32)
    if dest_acc == DestAccumulation.No:
        ends = truncate_to_dest_width(ends, formats.input_format)
    ends = ends.to(format_dict[formats.output_format]).to(torch.float64)
    return ends.min(dim=0).values, ends.max(dim=0).values
