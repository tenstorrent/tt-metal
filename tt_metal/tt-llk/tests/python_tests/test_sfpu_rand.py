# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
rand SFPU test. Covers llk_sfpu/ckernel_sfpu_rand.h on Wormhole and Blackhole.

rand has no element-wise golden, so the test checks properties: every element of the
tile is a draw inside [from, from + scale], and the draws cover the value grid that the
Dest width and output format can represent. A store whose width does not match the Dest
mode leaves elements outside the range (issue #14213).
"""

import struct

import torch
from conftest import skip_for_quasar
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import RAND_RANGE, RAND_SEED

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

# (from, scale): a positive range, and a negative one where BF16 truncation moves toward
# the upper bound. All endpoints are exact in BF16.
RANGES = [(2.0, 1.0), (-3.0, 2.0)]

SEED = 0x12345678
# Outside every range, so an element the kernel never wrote fails the range check.
SENTINEL = 100.0
FACE_ELEMENTS = ELEMENTS_PER_TILE // 4
# Draws must reach this fraction of the values the grid can represent in the range.
MIN_COVERAGE = 0.8
# A face has only 256 draws, so collisions on a small grid leave fewer distinct values.
MIN_FACE_COVERAGE = MIN_COVERAGE / 2


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _valid_dest_acc(formats):
    if formats.input_format.is_32_bit():
        return [DestAccumulation.Yes]
    return get_valid_dest_accumulation_modes(formats)


def _bf16_values_in(low: float, high: float) -> int:
    grid = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    grid = grid.to(torch.float32)
    return int(((grid >= low) & (grid <= high)).sum())


def _expected_distinct(formats, dest_acc, low, high, count):
    bf16_grid = (
        dest_acc == DestAccumulation.No or formats.output_format == DataFormat.Float16_b
    )
    representable = _bf16_values_in(low, high) if bf16_grid else count
    return min(representable, count)


@skip_for_quasar
@parametrize(
    formats=FORMATS,
    dest_acc=lambda formats: _valid_dest_acc(formats),
    rand_range=RANGES,
)
def test_sfpu_rand(formats, dest_acc, rand_range):
    rand_from, rand_scale = rand_range
    low, high = rand_from, rand_from + rand_scale

    torch_format = format_dict[formats.input_format]
    src_A = torch.full((ELEMENTS_PER_TILE,), SENTINEL, dtype=torch_format)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    configuration = TestConfig(
        "sources/sfpu_rand_test.cpp",
        formats,
        templates=[
            RAND_RANGE(
                rand_from_bits=_bits(rand_from), rand_scale_bits=_bits(rand_scale)
            ),
            RAND_SEED(rand_seed=SEED),
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
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format]).to(
        torch.float32
    )

    outside = ~((res >= low) & (res <= high))
    assert not bool(outside.any()), (
        f"{int(outside.sum())}/{ELEMENTS_PER_TILE} elements outside [{low}, {high}], "
        f"first at flat offsets {outside.nonzero().flatten().tolist()[:16]}: "
        f"{res[outside][:16].tolist()}"
    )

    expected = _expected_distinct(formats, dest_acc, low, high, ELEMENTS_PER_TILE)
    distinct = int(torch.unique(res).numel())
    assert distinct >= MIN_COVERAGE * expected, (
        f"only {distinct} distinct draws in the tile; the grid admits {expected} "
        f"values in [{low}, {high}]"
    )

    faces = res.view(4, FACE_ELEMENTS)
    face_expected = _expected_distinct(formats, dest_acc, low, high, FACE_ELEMENTS)
    for face in range(4):
        face_distinct = int(torch.unique(faces[face]).numel())
        assert face_distinct >= MIN_FACE_COVERAGE * face_expected, (
            f"face {face} holds only {face_distinct} distinct draws "
            f"(grid admits {face_expected})"
        )
        for other in range(face):
            assert not torch.equal(
                faces[face], faces[other]
            ), f"faces {other} and {face} are identical"
