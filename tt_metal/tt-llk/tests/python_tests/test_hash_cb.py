# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone LLK test for the experimental SFPU-backed CB hash (23-bit FNV).
#
# Tests run on both Blackhole and Wormhole B0. Both architectures implement
# the same FNV23 algorithm and apply the same UnshuffleFP32 permutation on
# SFPLOAD INT32 (verified against WormholeB0 SFPLOAD.md and BlackholeA0
# SFPLOAD.md), so a single golden function covers both.
#
# STATUS: draft. Hardware-validated on BH; WH hardware validation pending.

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT, generate_input_dim

# --- 23-bit FNV constants (must match llk_math_hash_cb.h for both BH and WH) ---
FNV23_INIT = 0x1C9DC5
FNV23_PRIME = 0x000193
MASK23 = 0x7FFFFF

# Architectures that support the SFPU hash variant.
SFPU_HASH_ARCHS = {ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE}


def unshuffle_fp32(w: int) -> int:
    """Apply the Dst->LReg permutation that SFPLOAD INT32 performs on 32-bit loads.

    Both WH and BH SFPLOAD.md specify that Dst stores FP32 as
        [sign:1][man_hi:7][exp:8][man_lo:16]
    and UnshuffleFP32 reorders on load to
        [sign:1][exp:8][man_hi:7][man_lo:16].

    This is a deterministic bit permutation applied to every raw u32 word that
    the SFPU hash kernel sees. The golden must apply the same permutation.
    """
    w &= 0xFFFFFFFF
    sign = (w >> 31) & 0x1
    man_hi = (w >> 24) & 0x7F  # bits 30..24
    exp = (w >> 16) & 0xFF  # bits 23..16
    man_lo = w & 0xFFFF
    return (sign << 31) | (exp << 23) | (man_hi << 16) | man_lo


def fnv23_lanewise_golden(words_u32) -> int:
    """Python model of the SFPU FNV23 hash — same for BH and WH.

    words_u32: iterable of u32 values in the order the SFPU loads them
               (row-major within a tile, tile-major across tiles).

    Each word is UnshuffleFP32'd (matching SFPLOAD INT32 behaviour), then
    accumulated per-lane with the FNV23 multiply. The 32 per-lane accumulators
    are XOR-folded to produce a single 23-bit hash.
    """
    lanes = [FNV23_INIT] * 32
    for i, w in enumerate(words_u32):
        shuffled = unshuffle_fp32(int(w))
        lane = i % 32
        lanes[lane] = ((lanes[lane] ^ shuffled) * FNV23_PRIME) & MASK23
    combined = 0
    for h in lanes:
        combined ^= h
    return combined & MASK23


def _skip_if_unsupported():
    if TestConfig.CHIP_ARCH not in SFPU_HASH_ARCHS:
        pytest.skip(f"SFPU hash not supported on {TestConfig.CHIP_ARCH}")


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1, 2, 4],
    seed=[42, 123, 999],
    dest_acc=[DestAccumulation.Yes],
)
def test_hash_cb_sfpu(formats, num_tiles, seed, dest_acc):
    _skip_if_unsupported()

    torch.manual_seed(seed)

    # Input: num_tiles stacked 32x32 int32 tiles.
    input_dimensions = [32 * num_tiles, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    # Compute the golden hash in Python over the raw u32 words in row-major order.
    words = src_A.flatten().to(torch.int32).cpu().numpy().astype("uint32").tolist()
    golden_hash = fnv23_lanewise_golden(words)

    configuration = TestConfig(
        "sources/hash_cb_test.cpp",
        formats,
        templates=[generate_input_dim(input_dimensions, input_dimensions)],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,  # single hash tile
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    res_from_L1 = configuration.run().result

    # The first u32 of the packed output tile carries the reduced 23-bit hash.
    torch_dtype = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_dtype).flatten()
    hw_hash = int(res_tensor[0].item()) & MASK23

    assert (
        hw_hash == golden_hash
    ), f"SFPU hash {hex(hw_hash)} != golden {hex(golden_hash)}"


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[2],
    seed=[7],
    dest_acc=[DestAccumulation.Yes],
)
def test_hash_cb_sfpu_determinism(formats, num_tiles, seed, dest_acc):
    """Two kernel runs on the same input must produce bit-identical hashes."""
    _skip_if_unsupported()

    torch.manual_seed(seed)
    input_dimensions = [32 * num_tiles, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    def run_once():
        configuration = TestConfig(
            "sources/hash_cb_test.cpp",
            formats,
            templates=[generate_input_dim(input_dimensions, input_dimensions)],
            runtimes=[TILE_COUNT(tile_cnt_A)],
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_B,
                tile_count_res=1,
            ),
            dest_acc=dest_acc,
            unpack_to_dest=formats.input_format.is_32_bit(),
        )
        res = configuration.run().result
        torch_dtype = format_dict[formats.output_format]
        return int(torch.tensor(res, dtype=torch_dtype).flatten()[0].item()) & MASK23

    h1 = run_once()
    h2 = run_once()
    assert h1 == h2, f"SFPU hash non-deterministic across runs: {hex(h1)} vs {hex(h2)}"
