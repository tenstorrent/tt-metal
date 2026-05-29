# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone LLK test for the SFPU-backed CB hash (23-bit FNV23).
#
# Tests run on both Blackhole and Wormhole B0. The kernel exercises SFPLOAD,
# SFPMUL24 (BH) / shift-and-add (WH), SFPXOR, and SFPSHFT2 reduction, then
# MATH publishes the reduced hash to a fixed slot in the MEM_LLK_DEBUG L1
# region. The UNPACK thread polls the ready flag, reads the hash back out of
# L1, and writes it into buffer_Res[0] so this test can pick it up via the
# usual result-buffer path.
#
# The tests verify:
#   1. The kernel runs without asserts and produces a valid 23-bit hash.
#   2. Two runs on the same input produce bit-identical results (determinism).
#
# A Python golden is intentionally omitted: DEST face ordering, the
# UnshuffleFP32 permutation on SFPLOAD INT32, and SFPMUL24 / shift-and-add
# truncation semantics need hardware calibration before a golden can be
# trusted. The determinism test still validates the property that matters
# for bisection debugging.
#
# STATUS: WH SFPU sequence is hardware-validated (WH B0 n150) via the
# sibling tt-metal gtest at tests/tt_metal/tt_metal/llk/test_cb_hash.cpp.
# BH SFPU sequence is still pending hardware bring-up.

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

MASK23 = 0x7FFFFF

# Architectures that support the SFPU hash variant.
SFPU_HASH_ARCHS = {ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE}


def _skip_if_unsupported():
    if TestConfig.CHIP_ARCH not in SFPU_HASH_ARCHS:
        pytest.skip(f"SFPU hash not supported on {TestConfig.CHIP_ARCH}")


def _run_hash_kernel(
    formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
):
    """Run the SFPU hash kernel and return the 23-bit hash."""
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
    torch_dtype = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_dtype).flatten()
    return int(res_tensor[0].item())


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1],
    seed=[42],
    dest_acc=[DestAccumulation.Yes],
)
def test_hash_cb_sfpu(formats, num_tiles, seed, dest_acc):
    """Verify the SFPU hash kernel runs and produces a valid 23-bit result."""
    _skip_if_unsupported()

    torch.manual_seed(seed)

    input_dimensions = [32 * num_tiles, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    hw_hash = _run_hash_kernel(
        formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
    )

    # Sanity: hash should be non-zero and fit in 23 bits.
    assert (
        hw_hash != 0
    ), "SFPU hash returned zero — kernel likely did not execute correctly"
    assert hw_hash <= MASK23, f"SFPU hash {hex(hw_hash)} exceeds 23-bit range"


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1],
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
    )

    h1 = _run_hash_kernel(
        formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
    )
    h2 = _run_hash_kernel(
        formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
    )
    assert h1 == h2, f"SFPU hash non-deterministic across runs: {hex(h1)} vs {hex(h2)}"
