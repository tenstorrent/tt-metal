# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone LLK test for the experimental SFPU-backed CB hash (23-bit FNV).
#
# STATUS: draft. The matching kernel source (sources/hash_cb_test.cpp) and the
# LLK-lib implementation (tt_llk_blackhole/llk_lib/experimental/llk_math_hash_cb.h)
# have not yet been hardware-validated. This pytest encodes the intended golden
# so that the implementation can be iterated to match once someone runs it on a
# Blackhole device.

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

# --- 23-bit FNV constants (match llk_math_hash_cb.h) ---
FNV23_INIT = 0x1C9DC5
FNV23_PRIME = 0x000193
MASK23 = 0x7FFFFF


def unshuffle_fp32(w: int) -> int:
    """Apply the Dst->LReg permutation that SFPLOAD performs on 32-bit loads.

    Per BlackholeA0 SFPLOAD.md (lines 194-202), Dst stores FP32 as
        [sign:1][man_hi:7][exp:8][man_lo:16]
    and UnshuffleFP32 reorders on load to
        [sign:1][exp:8][man_hi:7][man_lo:16].

    For an opaque 32-bit word, this is a deterministic bit permutation that
    the SFPU-side FNV23 will see. The golden must apply the same permutation.
    """
    w &= 0xFFFFFFFF
    sign = (w >> 31) & 0x1
    man_hi = (w >> 24) & 0x7F  # bits 30..24
    exp = (w >> 16) & 0xFF  # bits 23..16
    man_lo = w & 0xFFFF
    return (sign << 31) | (exp << 23) | (man_hi << 16) | man_lo


def fnv23_lanewise_golden(words_u32) -> int:
    """NumPy / Python-level model of the SFPU FNV23 hash.

    words_u32: iterable of u32 values in the order the SFPU will load them
               (row-major within a tile, tile-major across tiles).
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


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1, 2, 4],
    seed=[42, 123, 999],
    dest_acc=[DestAccumulation.Yes],
)
def test_hash_cb_sfpu(formats, num_tiles, seed, dest_acc):
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("SFPU hash variant is Blackhole-only in this PR.")

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
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("SFPU hash variant is Blackhole-only in this PR.")

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
        )
        res = configuration.run().result
        torch_dtype = format_dict[formats.output_format]
        return int(torch.tensor(res, dtype=torch_dtype).flatten()[0].item()) & MASK23

    h1 = run_once()
    h2 = run_once()
    assert h1 == h2, f"SFPU hash non-deterministic across runs: {hex(h1)} vs {hex(h2)}"
