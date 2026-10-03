# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unary fmod / remainder with a non-power-of-two scalar divisor (Blackhole only).

The generic unary SFPU suite drives MathOperation.Fmod / Remainder with the divisor fixed at
2.0. A power of two has an exact reciprocal and an exact fp32 quotient, so that sweep never
exercises the part of ckernel_sfpu_fmod.h / ckernel_sfpu_remainder.h that does the work:
rounding the estimated quotient to an integer and correcting the residual. This module drives
sources/sfpu_fmod_scalar_test.cpp with the divisors ttnn users actually pass (3, 7, 1.5,
0.003, ...) and the host reciprocal computed the way ttnn computes it (fl(1/divisor)).

Contract asserted. fmod and remainder of two fp32 numbers are exactly representable in fp32
(the result is a multiple of ulp(divisor) smaller than |divisor|), so for every finite lane
whose true quotient |x/s| is below 2^24 the kernel must return the exact value, bit for bit:
the fp64 torch.fmod / torch.remainder of the fp32 operands is that exact value. Above 2^24 the
integer quotient no longer fits an fp32 mantissa and no fp32-only kernel can recover the
remainder, so those quotients are kept out of the populations here.

Zero sign. torch.remainder builds on fmod and gives a zero result the dividend's sign; the
kernel gives it the divisor's sign (copysgn(0, s)), unchanged from its earlier form. That
difference is invisible to every tolerance comparison and is deliberately not asserted:
remainder lanes are compared with the zero sign masked. fmod zeros are asserted exactly
(dividend's sign in both).

NaN. Every lane whose reference is NaN (NaN or +-inf dividend) must come back NaN; the sign of
that NaN is not asserted (fmod copies the dividend's sign onto it, remainder the divisor's).

Denormals. The SFPU flushes a denormal dividend to zero (measured on Blackhole for both this
kernel and the shift-truncate form it replaces), so those lanes expect +-0 (fmod: the dividend's
sign) rather than the denormal itself that torch returns.
"""

import math
import struct

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    SFPU_UNARY_SCALAR,
    SFPU_UNARY_THRESHOLD,
    VECTOR_MODE,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# fp32 in, fp32 out, 32-bit DEST: the only pipeline in which the exact fp32 result is
# observable. bf16 inputs are a subset of fp32 inputs and the bf16 pack rounding would only
# hide bits.
FORMATS = input_output_formats([DataFormat.Float32], same=True)

# Non-power-of-two divisors of both signs and very different magnitudes. 3 and 7 have short
# mantissas (exact multiples up to large n), 0.003 / 0.0029 / 3.14159 / 1e-3 use all 24 bits
# (the reciprocal is inexact and n*s rounds), 1.5 and 100 sit in between; -3 and -0.0029
# cover the divisor-sign fix-ups of remainder.
DIVISORS = (3.0, 7.0, 1.5, -3.0, 0.003, -0.0029, 3.14159, 1e-3, 100.0)

POPULATIONS = ("dense_q_below_2p24", "exact_multiples_and_neighbours", "specials")

QUOTIENT_LIMIT = 2.0**24


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _host_reciprocal_bits(divisor: float) -> int:
    """fl(1/divisor) in fp32, as ttnn's unary_op_utils.cpp computes it (1.0f / param0)."""
    return _bits(_f32(1.0 / _f32(divisor)))


def _population(name: str, divisor: float) -> torch.Tensor:
    """One tile of fp32 dividends for *divisor*; every finite lane has |x/s| < 2^24."""
    s = abs(_f32(divisor))
    g = torch.Generator().manual_seed(0x600D)
    if name == "dense_q_below_2p24":
        # log-uniform |q| in [2^-4, 2^24), both signs
        e = torch.empty(4 * ELEMENTS_PER_TILE, dtype=torch.float64).uniform_(
            -4.0, 24.0, generator=g
        )
        sign = torch.where(torch.rand(e.numel(), generator=g) < 0.5, -1.0, 1.0).double()
        x = (torch.exp2(e) * s * sign).float()
    elif name == "exact_multiples_and_neighbours":
        # n*s (fp64 product rounded to fp32) and its two fp32 neighbours: the neighbours are
        # the inputs one ULP below/above an exact multiple that a truncated quotient gets wrong.
        n = torch.arange(1, 2 * ELEMENTS_PER_TILE, dtype=torch.float64)
        n = torch.cat([n, -n])
        mult = (n * _f32(divisor)).float()
        below = torch.nextafter(mult, torch.zeros_like(mult))
        above = torch.nextafter(
            mult, torch.copysign(torch.full_like(mult, math.inf), mult)
        )
        x = torch.stack([mult, below, above], dim=1).reshape(-1)
    elif name == "specials":
        tiny = torch.finfo(torch.float32).tiny  # smallest normal
        sub = math.ldexp(1.0, -149)  # smallest subnormal
        vals = [
            0.0,
            -0.0,
            math.inf,
            -math.inf,
            math.nan,
            -math.nan,
            tiny,
            -tiny,
            sub,
            -sub,
            1e-40,
            -1e-40,
            1.0,
            -1.0,
            0.5,
            -0.5,
            s,
            -s,
            s / 2,
            -s / 2,
            2 * s,
            -2 * s,
            1.5 * s,
            -1.5 * s,
            (2**23 + 1) * s,
            -(2**23 + 1) * s,
            (2**24 - 2) * s,
            -(2**24 - 2) * s,
            (2**22 + 0.5) * s,
            -(2**22 + 0.5) * s,
            (2**23 + 0.5) * s,
            -(2**23 + 0.5) * s,
        ]
        x = torch.tensor(vals, dtype=torch.float64).float()
    else:
        raise ValueError(name)

    # keep only lanes whose fp32 quotient is inside the exact regime (or non-finite)
    q = x.double().abs() / s
    keep = ~torch.isfinite(x) | (q < QUOTIENT_LIMIT)
    x = x[keep][:ELEMENTS_PER_TILE]
    out = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)
    out[: x.numel()] = x
    return out


def _reference(mathop: MathOperation, x: torch.Tensor, divisor: float) -> torch.Tensor:
    s = torch.tensor(_f32(divisor), dtype=torch.float64)
    fn = torch.fmod if mathop == MathOperation.Fmod else torch.remainder
    ref = fn(x.double(), s).float()
    # SFPU flush-to-zero: a denormal dividend reads as +-0, so its remainder is +-0.
    denormal = (x != 0) & (x.abs() < torch.finfo(torch.float32).tiny)
    ref[denormal] = torch.copysign(torch.zeros_like(x[denormal]), x[denormal])
    return ref


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.Yes],
    mathop=[MathOperation.Fmod, MathOperation.Remainder],
    divisor=list(DIVISORS),
    population=runtime(list(POPULATIONS)),
)
def test_sfpu_fmod_scalar(formats, dest_acc, mathop, divisor, population):
    x = _population(population, divisor)
    ref = _reference(mathop, x, divisor)

    configuration = TestConfig(
        "sources/sfpu_fmod_scalar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(divisor)),
            SFPU_UNARY_THRESHOLD(_host_reciprocal_bits(divisor)),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            x,
            formats.input_format,
            torch.zeros_like(x),
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

    res = configuration.run().result[:ELEMENTS_PER_TILE]
    out = (
        torch.tensor(res, dtype=format_dict[formats.output_format])
        .flatten()
        .to(torch.float32)
    )

    ref_nan = torch.isnan(ref)
    assert torch.isnan(out[ref_nan]).all(), (
        f"{mathop.name} s={divisor}: NaN expected where the dividend is NaN/inf, got "
        f"{out[ref_nan & ~torch.isnan(out)].tolist()[:8]}"
    )

    fin = ~ref_nan
    out_bits = out.view(torch.int32)
    ref_bits = ref.view(torch.int32)
    if mathop == MathOperation.Remainder:
        # zero sign follows the divisor in the kernel and the dividend in torch: mask it
        zero = (ref == 0) & (out == 0)
        fin = fin & ~zero
    bad = fin & (out_bits != ref_bits)
    n_bad = int(bad.sum())
    if n_bad:
        idx = torch.nonzero(bad).reshape(-1)[:8].tolist()
        detail = ", ".join(
            f"x={x[i].item():.9g}: got {out[i].item():.9g} want {ref[i].item():.9g}"
            for i in idx
        )
        pytest.fail(
            f"{mathop.name} s={divisor} [{population}]: {n_bad} inexact lanes: {detail}"
        )
    # The defining contract, stated separately. One lane is exempt: remainder(x, s) for x just below
    # zero (e.g. -FLT_MIN) is |s| - |x|, which rounds to |s| itself in fp32; torch returns |s| there
    # too, and the bitwise comparison above has already matched it.
    s = abs(_f32(divisor))
    in_range = (out[fin].abs() < s) | (ref[fin].abs() >= s)
    assert bool(
        in_range.all()
    ), f"{mathop.name} s={divisor}: |result| >= |divisor| on some lane"
