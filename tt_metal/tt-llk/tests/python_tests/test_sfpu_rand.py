# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

#
# Regression test for the SFPU `rand` generator store width.
#
# `rand` fills a Dest tile with uniform floats in [from, from + scale). Its
# SFPSTORE must write in the Dest's *configured* width: a store that writes the
# wrong Dest view (e.g. a 16-bit store into a 32-bit Dest) leaves half of the
# tile unwritten -- historically "512 correct, 512 zero". This test drives rand
# in both fp32-dest-accumulation modes and asserts the whole tile is filled with
# in-range values, so a store-width regression is caught on-card.
#
# There is no host model of the Tensix PRNG, so the check is a distribution
# invariant (full fill + range), not an exact golden.

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import APPROX_MODE

# Must match the constexpr values in sources/sfpu_rand_test.cpp.
RAND_FROM = 1024.0
RAND_SCALE = 256.0
TILE_ELEMENTS = 1024


def _run_sfpu_rand(formats, dest_acc, input_dimensions=[32, 32]):
    num_elements = input_dimensions[0] * input_dimensions[1]

    # Zeroed input: the kernel copies it into Dest before rand overwrites, so any
    # Dest row the generator fails to write stays 0 and fails the range check.
    src_A = torch.zeros(num_elements, dtype=torch.float32)
    src_B = torch.zeros(num_elements, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_rand_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[],
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[:TILE_ELEMENTS]
    assert len(res_from_L1) == TILE_ELEMENTS, "Unexpected result tile length"

    torch_format = format_dict[formats.output_format]
    res = torch.tensor(res_from_L1, dtype=torch_format).to(torch.float32)

    lo = RAND_FROM
    hi = RAND_FROM + RAND_SCALE
    # bf16 rounding near 1024-1280 is ~8 ULP; fp32 is exact.
    tol = 8.0 if formats.output_format == DataFormat.Float16_b else 1.0

    num_zeros = int((res == 0).sum().item())
    tile_min = res.min().item()
    tile_max = res.max().item()

    # Whole tile must be filled with in-range values. A store-width regression
    # (the original bug) leaves ~half the tile at 0.0, which is far below `from`.
    assert num_zeros == 0, f"{num_zeros}/{TILE_ELEMENTS} elements are 0 (tile not fully written by rand)"
    assert tile_min >= lo - tol, f"min {tile_min} below range [{lo}, {hi}) (tol={tol})"
    assert tile_max <= hi + tol, f"max {tile_max} above range [{lo}, {hi}) (tol={tol})"
    # rand must produce a distribution, not a constant fill.
    assert res.std().item() > 0.0, "rand produced a constant tile"


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_rand(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 with dest_acc=No is not supported")
    if formats.input_format == DataFormat.Float16_b and dest_acc == DestAccumulation.Yes:
        pytest.skip("Float16_b with dest_acc=Yes is not supported")

    _run_sfpu_rand(formats, dest_acc)
