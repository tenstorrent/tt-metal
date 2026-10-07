# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
FP32 accuracy and range coverage for the binary power kernel.

Drives _sfpu_binary_power_<true> through calculate_rpow, with the same kernel source as
test_sfpu_binary_pow_zero_base.py: the base is a compile-time scalar (SFPU_UNARY_SCALAR) and
the exponent tile comes from DEST.

    accuracy           bases 0.9, 1.1, 2.5 and 10 with |y * log2 x| up to 125. The kernel
                       computes 2**(y * log2|x|), so an error in log2|x| reaches the result
                       multiplied by |y|; for base 0.9 these exponents reach |y| ~ 820.
                       Max 2 ULP against float64 pow rounded once to fp32.
    huge exponents     |y| = 1e35 and FLT_MAX: pow(1, y) = 1, and every other base gives 0 or
                       +inf (#55129).
    underflow          results below the smallest normal give +0, not NaN (#57446),
                       including pow(0.16287974, 48.508072), where y * log2 x sits just
                       below -127.

ULP is helpers.ulp's distance in representable fp32 values, so a wrong sign fails. Only finite
normal results are counted in the accuracy test: subnormal results flush to zero on store.
"""

import math
import struct

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)
from helpers.ulp import within_ulp

pytestmark = [skip_for_quasar]

# FP32 in and out. DestAccumulation.Yes is the only valid mode for a 32-bit input.
FORMATS = input_output_formats([DataFormat.Float32], same=True)

ELEMENTS_PER_TILE = 1024
SMALLEST_NORMAL = 1.1754943508222875e-38
FLT_MAX = 3.4028234663852886e38


def _bits(value: float) -> int:
    """Raw fp32 bit pattern"""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _fp32(value: float) -> float:
    """value rounded to fp32, as the kernel sees it"""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _raw_bits(t: torch.Tensor) -> torch.Tensor:
    """fp32 bit patterns as int64, so that +0 and -0 differ."""
    return t.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)


def _run_rpow(formats, base, exponents):
    """base**exponents on the device, one tile, fp32."""
    src_A = exponents.to(torch.float32).contiguous()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_binary_pow_scalar_base_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(base)),
            VECTOR_MODE(VectorMode.RC),
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
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    return torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)


def _golden(base, exponents):
    """torch pow in float64 on the fp32 inputs, rounded once to fp32."""
    base64 = torch.tensor(_fp32(base), dtype=torch.float64)
    return torch.pow(base64, exponents.to(torch.float64)).to(torch.float32)


@parametrize(formats=FORMATS, base=[0.9, 1.1, 2.5, 10.0])
def test_sfpu_binary_pow_fp32_ulp(formats, base):
    """Max 2 ULP for |y * log2 x| <= 125."""
    lim = 125.0 / abs(math.log2(_fp32(base)))
    gen = torch.Generator().manual_seed(5)
    exponents = (
        torch.empty(ELEMENTS_PER_TILE, dtype=torch.float64)
        .uniform_(-lim, lim, generator=gen)
        .to(torch.float32)
    )

    device = _run_rpow(formats, base, exponents)
    golden = _golden(base, exponents)

    mask = torch.isfinite(golden) & (golden.abs() >= SMALLEST_NORMAL)
    assert int(mask.sum()) > ELEMENTS_PER_TILE // 2
    ok, message = within_ulp(
        golden, device, max_ulp=2, fmt=DataFormat.Float32, mask=mask
    )
    assert ok, f"pow({base}, y): {message}"


@parametrize(formats=FORMATS, base=[1.0, 0.5, 2.0])
def test_sfpu_binary_pow_fp32_huge_exponent(formats, base):
    """|y| = 1e35 and FLT_MAX: 1 for base 1, else 0 or +inf as IEEE 754 pow (#55129)."""
    values = [1e35, -1e35, FLT_MAX, -FLT_MAX]
    exponents = torch.tensor(values, dtype=torch.float32).repeat(
        ELEMENTS_PER_TILE // len(values)
    )

    device = _run_rpow(formats, base, exponents)
    golden = _golden(base, exponents)

    for i in range(len(values)):
        got, want = device[i :: len(values)], golden[i :: len(values)]
        assert (_raw_bits(got) == _raw_bits(want)).all(), (
            f"pow({base}, {exponents[i].item():g}) = {got[0].item()!r}, "
            f"expected {want[0].item()!r}"
        )


@parametrize(
    formats=FORMATS,
    base_exponent=[
        (0.16287973523139954, 48.50807189941406),
        (0.1, 1000.0),
        (0.003813433460891247, 291.2920227050781),
        (5.8008667314091156e-11, 115.76145935058594),
    ],
)
def test_sfpu_binary_pow_fp32_underflow(formats, base_exponent):
    """A result below the smallest normal fp32 is +0, not NaN (#57446)."""
    base, exponent = base_exponent
    # The issue's exponent first, then larger ones: every result is below the smallest normal.
    exponents = torch.linspace(1.0, 4.0, ELEMENTS_PER_TILE, dtype=torch.float64)
    exponents = (exponents * exponent).to(torch.float32)
    exponents[0] = exponent
    assert (_golden(base, exponents).abs() < SMALLEST_NORMAL).all()

    device = _run_rpow(formats, base, exponents)

    bad = torch.nonzero(_raw_bits(device) != 0).flatten()
    assert bad.numel() == 0, (
        f"{bad.numel()}/{ELEMENTS_PER_TILE} lanes are not +0, e.g. "
        f"pow({base}, {exponents[bad[0]].item()!r}) = {device[bad[0]].item()!r}"
    )
