# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stimuli and result bounds for the scalar PROD SFPU reduce, shared by the WH/BH and Quasar tests."""

import math

import torch

from .format_config import InputOutputFormat
from .golden_generators import ELEMENTS_PER_TILE, TILE_DIM, truncate_to_dest_width
from .llk_params import DestAccumulation, format_dict

PROD_DIMENSIONS = [[2 * TILE_DIM, 2 * TILE_DIM]]
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

# Specials classes, each planted in an otherwise +-1 tile. overflow / underflow hold
# PROD_SPECIAL_RUN 2s or 0.5s among +1s: 2**+-160 leaves fp32's range.
PROD_SPECIAL_CLASSES = [
    "zero",
    "pos_inf",
    "neg_inf",
    "inf_times_zero",
    "nan",
    "overflow",
    "underflow",
]
PROD_SPECIAL_RUN = 160


def generate_prod_stimuli(stimulus: str, tile_cnt: int) -> torch.Tensor:
    """``pow2``: +-1, +-2 and +-0.5 only, so every product is exact; ``near_one``: random signs on
    magnitudes in (NEAR_ONE_LOW, NEAR_ONE_HIGH), so the fp32 multiply chain rounds."""
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
    """Bounds on each tile's element [0][0]: the exact product, widened by PROD_FP32_CHAIN_RTOL for
    near_one, then narrowed the way the device stores it (a 16-bit Dest store truncates toward zero).
    """
    exact = src.to(torch.float64).view(tile_cnt, ELEMENTS_PER_TILE).prod(dim=1)
    rtol = 0.0 if stimulus == "pow2" else PROD_FP32_CHAIN_RTOL
    ends = torch.stack((exact * (1 - rtol), exact * (1 + rtol))).to(torch.float32)
    if dest_acc == DestAccumulation.No:
        ends = truncate_to_dest_width(ends, formats.input_format)
    ends = ends.to(format_dict[formats.output_format]).to(torch.float64)
    return ends.min(dim=0).values, ends.max(dim=0).values


def generate_prod_special_tile(special: str) -> tuple[torch.Tensor, float]:
    """One tile for a PROD_SPECIAL_CLASSES entry, and the product the device must store."""
    tile = torch.ones(ELEMENTS_PER_TILE, dtype=torch.float64)
    if special in ("overflow", "underflow"):
        tile[:PROD_SPECIAL_RUN] = 2.0 if special == "overflow" else 0.5
        return tile, math.inf if special == "overflow" else 0.0

    gen = torch.Generator().manual_seed(PROD_STIMULUS_SEED)
    tile = torch.where(
        torch.rand(ELEMENTS_PER_TILE, generator=gen) < 0.5, -1.0, 1.0
    ).to(torch.float64)
    first, second = torch.randperm(ELEMENTS_PER_TILE, generator=gen)[:2].tolist()
    finite_sign = tile.prod().item() * tile[first].item()
    if special == "zero":
        tile[first] = 0.0
        return tile, 0.0
    if special in ("pos_inf", "neg_inf"):
        tile[first] = math.inf if special == "pos_inf" else -math.inf
        return tile, tile[first].item() * finite_sign
    if special == "inf_times_zero":
        tile[first] = math.inf
        tile[second] = 0.0
        return tile, math.nan
    if special == "nan":
        tile[first] = math.nan
        return tile, math.nan
    raise ValueError(f"unknown PROD special class {special!r}")


def prod_tile_results(res_from_L1, tile_cnt: int, output_format) -> torch.Tensor:
    """Element [0][0] of every result tile - the first datum of its face 0 - as float64."""
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[output_format])
    return res_tensor.view(tile_cnt, ELEMENTS_PER_TILE)[:, 0].to(torch.float64)


def check_prod_result(
    res_from_L1,
    src: torch.Tensor,
    tile_cnt: int,
    stimulus: str,
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
) -> None:
    """Assert every tile's element [0][0] lies inside prod_result_bounds."""
    lower, upper = prod_result_bounds(src, tile_cnt, stimulus, formats, dest_acc)
    result = prod_tile_results(res_from_L1, tile_cnt, formats.output_format)
    assert torch.all((result >= lower) & (result <= upper)), (
        f"scalar PROD mismatch ({stimulus}): device={result.tolist()} "
        f"allowed=[{lower.tolist()}, {upper.tolist()}]"
    )
