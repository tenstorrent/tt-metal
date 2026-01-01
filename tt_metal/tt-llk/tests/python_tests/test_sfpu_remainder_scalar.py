# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import struct

import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, VectorMode
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = [skip_for_quasar]

ELEMENTS_PER_TILE = 1024
FORMATS = input_output_formats([DataFormat.Float32], same=True)[0]


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=lambda s: "pos" if s > 0 else "neg")
@pytest.mark.parametrize("divisor", [3.0, 7.0, 1.5, -3.0])
def test_sfpu_remainder_scalar_just_below_exact_multiple(divisor, sign):
    n = torch.arange(1, ELEMENTS_PER_TILE + 1, dtype=torch.float64)
    exact_multiples = (n * abs(divisor)).to(torch.float32)
    src_A = (
        sign * torch.nextafter(exact_multiples, torch.zeros_like(exact_multiples))
    ).contiguous()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_remainder_scalar_test.cpp",
        FORMATS,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(divisor)),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
    )
    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE], dtype=torch.float32
    )

    expected = torch.remainder(src_A.double(), divisor).float()
    assert torch.allclose(res, expected, atol=1e-3, rtol=0), (
        f"remainder(x, {divisor}): {int((~torch.isclose(res, expected, atol=1e-3, rtol=0)).sum())} lanes off, "
        f"e.g. x={src_A[0].item()!r} gave {res[0].item()!r}, expected {expected[0].item()!r}"
    )
    assert torch.all(res.abs() < abs(divisor))
    assert torch.all((res == 0) | (torch.sign(res) == math.copysign(1.0, divisor)))


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=lambda s: "pos" if s > 0 else "neg")
@pytest.mark.parametrize("divisor", [0.1, 0.3])
def test_sfpu_remainder_scalar_rounded_multiples(divisor, sign):
    divisor_f32 = torch.tensor(divisor, dtype=torch.float32)
    n = torch.arange(1, ELEMENTS_PER_TILE + 1, dtype=torch.float64)
    src_A = (sign * n * divisor_f32.double()).to(torch.float32).contiguous()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_remainder_scalar_test.cpp",
        FORMATS,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(divisor)),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
    )
    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE], dtype=torch.float32
    )

    expected = torch.remainder(src_A, divisor_f32)
    off = (res - expected).abs() > 1e-3 * divisor
    assert not off.any(), (
        f"remainder(x, {divisor}): {int(off.sum())} of {ELEMENTS_PER_TILE} rounded multiples off, "
        f"e.g. x={src_A[off][0].item()!r} gave {res[off][0].item()!r}, expected {expected[off][0].item()!r}"
    )
