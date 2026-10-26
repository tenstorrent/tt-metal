# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Accuracy of the generic binary SFPU power, calculate_sfpu_binary<..., BinaryOp::POW, ...>
(MathOperation.SfpuElwpow), against a float64 reference.

The operand pairs are chosen so that ln(base) is large: power-of-two bases 2**k with
|k| up to 60, and bases drawn log-uniformly over [2**-40, 2**40]. Those are the inputs
where the old Blackhole body went wrong. It built ln(base) on the fp16-rounded
ln(2) = 0.692871 and exponentiated with the quadratic _sfpu_exp_ and repeated squaring,
so pow(2**20, 1) came out 4.0% low and pow(2**-60, 2) 28% high (tt-llk#1701 item 7).

Every pair keeps |log2(result)| <= 120, so the reference is a finite normal fp32 value
and no lane lands on the overflow or underflow knee.
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
    reason="Only the Blackhole generic BinaryOp::POW was fixed; the Wormhole copy still "
    "uses ln(2) = 0.692871 and _sfpu_exp_ (tenstorrent/tt-llk#1701 item 7)",
)

ELEMENTS_PER_TILE = 1024
# Tile pairs per variant: tile 2p holds the bases, tile 2p + 1 the exponents, and the
# kernel writes base**exponent over tile 2p.
TILE_PAIRS = 4
LANES = TILE_PAIRS * ELEMENTS_PER_TILE

# Worst relative error accepted against the float64 reference. Measured on Blackhole
# (p100a, 2026-10-08) over these populations: Float32 1.9e-07 (log-uniform) and 1.7e-08
# (powers of two), Float16_b 8.2e-03 (log-uniform, about two bf16 ULP, from the exp_21f body
# a bf16 Dest selects). Before the fix the same populations were off by up to 1.18.
MAX_REL_ERROR = {
    DataFormat.Float32: 1e-6,
    DataFormat.Float16_b: 2.0**-6,
}

_LOG2_RESULT_LIMIT = 120


def _powers_of_two():
    """Every (2**k, y) with k in [-60, 60], y in a fixed set and |k * y| within the limit."""
    ks = torch.arange(-60, 61, dtype=torch.float64)
    ys = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5], dtype=torch.float64)
    kk, yy = torch.meshgrid(ks, ys, indexing="ij")
    kk, yy = kk.flatten(), yy.flatten()
    keep = (kk * yy).abs() <= _LOG2_RESULT_LIMIT
    kk, yy = kk[keep], yy[keep]
    idx = torch.arange(LANES) % kk.numel()
    return torch.pow(2.0, kk[idx]), yy[idx]


def _log_uniform():
    """Bases 2**u, u uniform on [-40, 40], exponents uniform on [-3, 3], seeded."""
    gen = torch.Generator().manual_seed(1701)
    u = torch.rand(LANES, generator=gen, dtype=torch.float64) * 80 - 40
    y = torch.rand(LANES, generator=gen, dtype=torch.float64) * 6 - 3
    over = (u * y).abs() > _LOG2_RESULT_LIMIT
    y = torch.where(over, y * _LOG2_RESULT_LIMIT / (u * y).abs(), y)
    return torch.pow(2.0, u), y


_POPULATIONS = {
    "powers_of_two": _powers_of_two,
    "log_uniform": _log_uniform,
}


def _run_pow(formats, dest_acc, base, exponent):
    """Run base**exponent on the device and return it as float32, one value per lane."""
    torch_format = format_dict[formats.input_format]
    tiles = []
    for p in range(TILE_PAIRS):
        lanes = slice(p * ELEMENTS_PER_TILE, (p + 1) * ELEMENTS_PER_TILE)
        tiles += [base[lanes], exponent[lanes]]
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
            MATH_OP(mathop=MathOperation.SfpuElwpow),
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
    formats=input_output_formats([DataFormat.Float32, DataFormat.Float16_b], same=True),
    population=runtime(list(_POPULATIONS)),
)
def test_sfpu_binary_pow_accuracy(formats, population):
    # A 32-bit input needs a 32-bit Dest; bf16 runs with a 16-bit Dest, the
    # configuration a plain bf16 kernel uses.
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format == DataFormat.Float32
        else DestAccumulation.No
    )
    base, exponent = _POPULATIONS[population]()

    # The reference sees the operands the device sees, after the input format rounds them.
    torch_format = format_dict[formats.input_format]
    base = base.to(torch_format).to(torch.float64)
    exponent = exponent.to(torch_format).to(torch.float64)

    device = _run_pow(formats, dest_acc, base, exponent).to(torch.float64)
    golden = torch.pow(base, exponent)

    rel = ((device - golden) / golden).abs()
    rel = torch.where(torch.isnan(rel), torch.full_like(rel, float("inf")), rel)
    worst = int(rel.argmax())
    bound = MAX_REL_ERROR[formats.output_format]
    assert rel[worst] <= bound, (
        f"{int((rel > bound).sum())}/{LANES} lanes exceed relative error {bound:g}; worst "
        f"pow({base[worst].item():.9g}, {exponent[worst].item():g}) = "
        f"{device[worst].item():.9g}, expected {golden[worst].item():.9g} "
        f"(relative error {rel[worst].item():.3g})"
    )
