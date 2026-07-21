# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).

Exercises the experimental ``mul_reduce_scalar`` LLKs
(``llk_math_mul_reduce_scalar.h`` / ``llk_unpack_mul_reduce_scalar.h``):

    result = sum_over_all_tiles_and_elements(A * B)

stored in element ``[0]`` of the output tile. Per the op's contract (Compute
API doc + on-silicon gtest), only element ``[0]`` is defined — the packer's
other lanes are unspecified — so the test validates the reduced scalar alone.

Kernel B is held at 1.0 (matching the on-silicon gtest and the
``fuser_config/fpu_reduce_scalar.yaml`` recipe), so the multiply reduces to
``sum(A)``. Coverage: bf16 (num_tiles up to 8, the DEST half-sync capacity)
plus native fp32 DEST (up to 4 tiles), for HiFi2/HiFi4.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import MATH_FIDELITY, TILE_COUNT
from helpers.utils import tolerances

# 32x32 bf16 tile: 4 faces of 16x16.
ELEMENTS_PER_TILE = 1024
TILE_DIMENSIONS = [32, 32]

# Inputs are always bf16; only the DEST/output precision varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]


def _dest_acc(output_format):
    """Native fp32 DEST is required whenever the output is Float32."""
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


def _num_tiles_for_format(formats):
    """DEST half-sync holds 8 bf16 tiles or 4 fp32 tiles; every multiply-phase
    product must be resident before the reduce phase consumes it."""
    return (
        [1, 2, 3, 4]
        if _dest_acc(formats.output_format) == DestAccumulation.Yes
        else [1, 2, 3, 7, 8]
    )


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    num_tiles=_num_tiles_for_format,
)
def test_mul_reduce_scalar(formats, math_fidelity, num_tiles):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("mul_reduce_scalar is a Blackhole-only experimental LLK")

    dest_acc = _dest_acc(formats.output_format)
    input_dimensions = [num_tiles * TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]

    # A ~ U[0, 1] mirrors the on-silicon gtest and keeps the accumulated sum
    # well inside bf16's dynamic range for the larger tile counts.
    src_A, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    # B == 1.0 everywhere (matching the gtest and fpu_reduce_scalar.yaml):
    # A * B == A, so the fused op reduces to sum(A) over all tiles/elements.
    src_B = torch.ones(
        tile_cnt_B * ELEMENTS_PER_TILE, dtype=format_dict[formats.input_format]
    )

    # Golden mirrors the on-silicon reference (test_mul_reduce_scalar.cpp): the
    # element-wise product summed over every element of every tile, in fp32.
    golden_scalar = float(
        (src_A.to(torch.float32) * src_B.to(torch.float32)).sum().item()
    )

    configuration = TestConfig(
        "sources/mul_reduce_scalar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == ELEMENTS_PER_TILE
    ), f"Expected one {ELEMENTS_PER_TILE}-element output tile, got {len(res_from_L1)}"

    # The reduced scalar lives in element [0]; every other lane is unspecified.
    device_scalar = float(res_from_L1[0])
    tol = tolerances[formats.output_format]
    assert abs(device_scalar - golden_scalar) <= tol.atol + tol.rtol * abs(
        golden_scalar
    ), (
        f"mul_reduce_scalar mismatch: device={device_scalar} golden={golden_scalar} "
        f"(num_tiles={num_tiles}, fidelity={math_fidelity.name})"
    )
