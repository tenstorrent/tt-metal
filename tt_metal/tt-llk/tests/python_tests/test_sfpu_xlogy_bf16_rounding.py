# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Rounding of XLOGY, calculate_sfpu_binary<..., BinaryOp::XLOGY, ...>, into a 16-bit (bf16) Dest.

xlogy(x, y) = x * ln(y). Through a bf16 Dest the old Blackhole body stored y, ran the in-place
log body, which truncated ln(y) to bf16 on its SFPSTORE, reloaded it, multiplied and truncated
the product on the final store. Both narrowings round toward zero: on Blackhole, xlogy(1, 3)
came out 1.09375 instead of the round-to-nearest-even 1.1015625, and over every positive normal
bf16 y with a random bf16 x only 13.8% of lanes matched the correctly rounded result, with a
mean error of -1.0 ULP toward zero (tt-llk#1701 item 17).

The reference is x * ln(y) in float64, rounded to nearest even into bf16. The kernel's ln is a
cubic, so a few lanes stay off by its approximation error; the test bounds how many and that
the error carries no sign bias.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BROADCAST_TYPE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    SFPU_BCAST_DIM,
    TILE_COUNT,
    generate_input_dim,
)

pytestmark = pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.BLACKHOLE,
    reason="Only the Blackhole XLOGY rounds a bf16 Dest to nearest even; the Wormhole copy "
    "still truncates twice (tenstorrent/tt-llk#1701 item 17)",
)

ELEMENTS_PER_TILE = 1024
# Tile pairs per variant: tile 2p holds x, tile 2p + 1 holds y, and the kernel writes
# xlogy(x, y) over tile 2p. 32 pairs cover all 32512 positive normal bf16 y once.
TILE_PAIRS = 32
LANES = TILE_PAIRS * ELEMENTS_PER_TILE

# Measured on Blackhole (p100a, 2026-10-08) with a bf16 Dest after the fix: 99.47% / 99.50%
# of lanes correctly rounded, mean signed error -0.001 / -0.000 ULP; identical, bit for bit,
# to the same inputs through a 32-bit Dest. Before the fix: 49.98% / 13.75%, -0.50 / -1.01 ULP.
MIN_CORRECTLY_ROUNDED = 0.99
MAX_MEAN_SIGNED_ULP = 0.02


def _positive_normal_bf16():
    """Every positive normal bf16 value, 0x0080 through 0x7F7F, as float64."""
    bits = torch.arange(0x0080, 0x7F80, dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16).to(torch.float64)


def _x_one():
    y = _positive_normal_bf16()
    y = y[torch.arange(LANES) % y.numel()]
    return torch.ones(LANES, dtype=torch.float64), y


def _x_random():
    y = _positive_normal_bf16()
    y = y[torch.arange(LANES) % y.numel()]
    gen = torch.Generator().manual_seed(1701)
    x = torch.rand(LANES, generator=gen, dtype=torch.float64) * 8 - 4
    return x.to(torch.bfloat16).to(torch.float64), y


_POPULATIONS = {
    "x_one": _x_one,
    "x_uniform_pm4": _x_random,
}


def _bf16_bits(values):
    return values.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF


def _from_bf16_bits(bits):
    return bits.to(torch.int16).view(torch.bfloat16).to(torch.float64)


def _round_to_bf16(values):
    """float64 -> nearest bf16, ties to even, without an intermediate fp32 rounding."""
    guess = _bf16_bits(values.to(torch.float32))
    best, best_bits = _from_bf16_bits(guess), guess
    best_err = (best - values).abs()
    for step in (-1, 1):
        bits = (guess + step) & 0xFFFF
        cand = _from_bf16_bits(bits)
        err = (cand - values).abs()
        better = torch.isfinite(cand) & (
            (err < best_err)
            | ((err == best_err) & (bits % 2 == 0) & (best_bits % 2 == 1))
        )
        best = torch.where(better, cand, best)
        best_bits = torch.where(better, bits, best_bits)
        best_err = torch.where(better, err, best_err)
    return best


def _ordinal(values):
    """bf16 values mapped to integers that step by one per representable bf16."""
    bits = _bf16_bits(values)
    return torch.where(bits >= 0x8000, -(bits & 0x7FFF), bits)


def _run_xlogy(formats, dest_acc, x, y):
    """Run xlogy(x, y) on the device and return it as float32, one value per lane."""
    torch_format = format_dict[formats.input_format]
    tiles = []
    for p in range(TILE_PAIRS):
        lanes = slice(p * ELEMENTS_PER_TILE, (p + 1) * ELEMENTS_PER_TILE)
        tiles += [x[lanes], y[lanes]]
    src_A = torch.cat(tiles).to(torch_format)
    src_B = torch.zeros_like(src_A)

    # 32 columns keep each 32x32 tile contiguous in the row-major buffer.
    dims = [64 * TILE_PAIRS, 32]
    tile_cnt = 2 * TILE_PAIRS
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, dims, [32, 32]
    )
    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            MATH_OP(mathop=MathOperation.SfpuXlogy),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            BROADCAST_TYPE(LlkBroadcastType.None_),
            SFPU_BCAST_DIM(LlkBroadcastType.None_),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(1),
            NUM_FACES(),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )
    result = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    ).flatten()
    return torch.cat(
        [
            result[2 * p * ELEMENTS_PER_TILE : (2 * p + 1) * ELEMENTS_PER_TILE]
            for p in range(TILE_PAIRS)
        ]
    ).to(torch.float32)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    population=runtime(list(_POPULATIONS)),
)
def test_sfpu_xlogy_bf16_dest_rounding(formats, population):
    x, y = _POPULATIONS[population]()
    device = _run_xlogy(formats, DestAccumulation.No, x, y).to(torch.float64)

    # The issue's worked example: ln 3 = 1.0986 is 1.1015625 rounded to nearest even and
    # 1.09375 truncated.
    one_three = (x == 1.0) & (y == 3.0)
    assert torch.all(
        device[one_three] == 1.1015625
    ), f"xlogy(1, 3) = {device[one_three][0].item()} through a bf16 Dest, expected 1.1015625"

    exact = x * torch.log(y)
    # x == 0 gives an exact zero, and a result under the smallest normal has no bf16 ULP grid
    # the ordinal comparison below can use.
    scored = (x != 0) & (exact.abs() >= 2.0**-126)
    golden = _round_to_bf16(exact[scored])
    ulp = _ordinal(device[scored]) - _ordinal(golden)
    # Positive when the device result is farther from zero than the correctly rounded one.
    signed = ulp * torch.sign(golden).to(ulp.dtype)

    correct = (ulp == 0).double().mean().item()
    bias = signed.double().mean().item()
    assert correct >= MIN_CORRECTLY_ROUNDED and abs(bias) <= MAX_MEAN_SIGNED_ULP, (
        f"{population}: {correct:.2%} of {int(scored.sum())} lanes correctly rounded "
        f"(need {MIN_CORRECTLY_ROUNDED:.0%}), mean signed error {bias:+.3f} ULP "
        f"(need |.| <= {MAX_MEAN_SIGNED_ULP}); "
        f"{int((signed < 0).sum())} lanes toward zero, {int((signed > 0).sum())} away"
    )
