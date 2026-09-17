# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Zero-base coverage for the binary power kernel(FP32 only).

Drives the entry point of
hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h:

    calculate_rpow<APPROX, ITERATIONS, fp32_dest_acc_en>(base)
        ->  dst = base ** dst

which calls _sfpu_binary_power_<fp32_dest_acc_en>. rpow is the entry point used here because
its base is a compile-time scalar and its exponent comes from DEST, so the base pins at a
zero while the tile sweeps the exponent classes.

Asserted, for base in {+0, -0} and per IEEE-754 pow:

    pow(+-0, NaN)     NaN
    pow(+-0, +-0)     1
    pow(+-0, +inf)    +0
    pow(+-0, -inf)    +inf
    pow(+-0, y > 0)   +-0     base sign kept only for odd integer y
    pow(+-0, y < 0)   +-inf   base sign kept only for odd integer y

Domain. Integer exponents stay small: parity comes off a vSMag16 conversion that saturates
outside +-32767, beyond which an odd exponent loses the result's sign (documented in the kernel).
"""

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

pytestmark = [skip_for_quasar]

# FP32 in and out. DestAccumulation.Yes is the only valid mode for a 32-bit input, so
# dest_acc is pinned rather than swept-and-skipped.
FORMATS = input_output_formats([DataFormat.Float32], same=True)

ELEMENTS_PER_TILE = 1024

INF = float("inf")
NAN = float("nan")
SMALLEST_NORMAL = 1.1754943508222875e-38

_EXPONENTS = (
    NAN,
    -NAN,
    INF,
    -INF,
    0.0,
    -0.0,
    1.0,
    2.0,  # positive even integer
    3.0,  # positive odd integer
    -1.0,  # negative odd integer
    -2.0,  # negative even integer
    0.5,  # positive non-integer
    -1.5,  # negative non-integer
    SMALLEST_NORMAL,
    -SMALLEST_NORMAL,
)

_BASES = (0.0, -0.0)


def _bits(value: float) -> int:
    """Raw fp32 bit pattern"""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _fmt(value: float) -> str:
    """Render a float so that a signed zero and a NaN sign are visible in the failure message."""
    sign = "-" if _bits(value) >> 31 else ""
    if value != value:
        return f"{sign}nan"
    if value == 0.0:
        return f"{sign}0.0"
    return f"{value:g}"


def _exponent_tile() -> torch.Tensor:
    """One tile cycling through the exponent classes so that every class lands in every face."""
    exponents = torch.tensor(_EXPONENTS, dtype=torch.float32)
    reps = ELEMENTS_PER_TILE // len(_EXPONENTS) + 1
    return exponents.repeat(reps)[:ELEMENTS_PER_TILE].contiguous()


def _build_pow_zero_base(formats, base):
    """Build one variant without running it, returning (configuration, exponents)."""
    src_A = _exponent_tile()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_binary_pow_zero_base_test.cpp",
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

    return configuration, src_A


def _finish_pow_zero_base(configuration, exponents, formats):
    """Run a prepared variant, returning (device_tensor, exponent_tensor) as fp32."""
    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    device = torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)
    return device, exponents.flatten()[:ELEMENTS_PER_TILE].to(torch.float32)


def _run_pow_zero_base(formats, base):
    """Compile+run one variant, returning (device_tensor, exponent_tensor) as fp32."""
    configuration, exponents = _build_pow_zero_base(formats, base)
    return _finish_pow_zero_base(configuration, exponents, formats)


@parametrize(
    formats=FORMATS,
    base=list(_BASES),
)
def test_sfpu_binary_pow_zero_base(formats, base):
    """base**exponent for base in {+0, -0}."""
    device, exponents = _run_pow_zero_base(formats, base)

    golden = torch.pow(torch.tensor(base, dtype=torch.float32), exponents)

    # Expected value here is 0, 1 or inf
    mismatches = []
    for i in range(ELEMENTS_PER_TILE):
        want, got = golden[i].item(), device[i].item()
        if want != want:
            if got == got:
                mismatches.append((exponents[i].item(), want, got))
        elif _bits(got) != _bits(want):
            mismatches.append((exponents[i].item(), want, got))

    if mismatches:
        classes = sorted(
            {(_fmt(e), _fmt(w), _fmt(g)) for e, w, g in mismatches},
            key=lambda row: row[0],
        )
        detail = "\n".join(
            f"  pow({_fmt(base)}, {exp}) = {got}, expected {want}"
            for exp, want, got in classes
        )
        raise AssertionError(
            f"{len(mismatches)}/{ELEMENTS_PER_TILE} lanes disagree with torch across "
            f"{len(classes)} exponent classes:\n{detail}"
        )
