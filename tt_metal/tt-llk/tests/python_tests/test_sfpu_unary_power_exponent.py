# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Exponent-class coverage for the unary power kernel, x**c with a compile-time scalar c.

Drives calculate_unary_power<APPROX, is_fp32_dest_acc_en, ITERATIONS>(exponent_bits) from
hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h through the standard unary
driver shape (sources/sfpu_unary_power_exponent_test.cpp), with the exponent carried by
SFPU_UNARY_SCALAR as raw fp32 bits -- exactly what power_tile(idst, param0) receives from ttnn.

The registry sweep (test_eltwise_unary_sfpu.py, MathOperation.UnaryPower) fixes c at 2.0, so it
reaches one of the classes the kernel dispatches on. This module sweeps c over all of them:
+-0, positive / negative x {even integer, odd integer, non-integer}, magnitudes beyond the
+-32767 range where the old vSMag16 parity test saturated, and +-Inf / NaN.

Base tile (65536 lanes): for a 16-bit Dst (bf16 input, or dest_acc=No where the unpacker
narrows the input to 16 bits) every bf16 bit pattern once, so the kernel sees exactly the
value the golden is computed from. For the fp32 path (Float32 input, dest_acc=Yes): the IEEE
specials, both zeros, denormals, +-1 and dense neighbourhoods of +-1, small integers and
log-uniform magnitudes across the whole normal range.

Asserted per lane against torch.pow evaluated in float64, on the kernel's documented contract:

    c == +-0                 -> 1 for every x, including +-Inf and NaN
    x == +-0, c > 0          -> 0
    x == +-0, c < 0 (or -Inf)-> NaN   (kernel contract, not IEEE, documented in the kernel)
    x < 0,  c non-integer    -> NaN   (+-Inf / NaN exponents classify as non-integer)
    x < 0,  c odd integer    -> negative result; even integer -> positive result
    finite x, |c| <= 8, result in the normal range -> relative error within the path's tolerance

Not asserted (outside the kernel's contract; the lanes are still run, so a bit-level A/B of the
raw result buffers between two trees covers them): +-Inf / NaN bases with c != 0, denormal
bases (pipeline FTZ), positive bases with a non-finite c, and the overflow / underflow tails
(|x**c| outside [2**-100, 2**100]). The tails are the unchanged log2/exp2 pipeline, whose
behaviour there is not IEEE (measured on Blackhole: the fp32 path returns NaN or a huge finite
value for a result below ~2**-127, the bf16 path a non-Inf pattern on overflow; see the review's
G02-12), so asserting them here would fail on the kernel as it is, not on the exponent handling.

One measured pipeline property is folded into the check: on a 16-bit Dst the kernel's NaN
constant reaches L1 as +Inf (0x7f80), on the fp32 Dst as NaN, on both the current and the
previous kernel. The NaN assertions accept +Inf on the 16-bit pipelines for that reason.
"""

import struct

import ml_dtypes
import numpy as np
import pytest
import torch
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
    generate_input_dim,
)

pytestmark = [skip_for_quasar]

INF = float("inf")
NAN = float("nan")

# 65536 lanes: every bf16 bit pattern once for the 16-bit paths.
INPUT_DIMENSIONS = [256, 256]
LANES = INPUT_DIMENSIONS[0] * INPUT_DIMENSIONS[1]

# name -> exponent. Names are the parametrize ids; the value is what the kernel is instantiated with.
EXPONENTS = {
    "zero": 0.0,
    "neg_zero": -0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "neg_one": -1.0,
    "neg_two": -2.0,
    "neg_three": -3.0,
    "half": 0.5,
    "neg_half": -0.5,
    "two_and_half": 2.5,
    "neg_one_and_half": -1.5,
    "milli": 1e-3,
    # Beyond the +-32767 range of the vSMag16 conversion the previous parity test saturated at.
    "even_32768": 32768.0,
    "odd_32769": 32769.0,
    "odd_2p23_plus_1": 8388609.0,
    "even_2p24": 16777216.0,
    "neg_even_65536": -65536.0,
    "even_1e30": 1e30,
    "inf": INF,
    "neg_inf": -INF,
    "nan": NAN,
}

# Exponents whose negative-base results the pre-classification Wormhole kernel gets wrong
# (vSMag16 saturation reads every |c| > 32767 as non-integer -> NaN). The Blackhole kernel
# classifies the exponent exactly.
_SATURATION_RANGE_EXPONENTS = {
    "even_32768",
    "odd_32769",
    "odd_2p23_plus_1",
    "even_2p24",
    "neg_even_65536",
    "even_1e30",
}

# Value check only where the kernel's log2/exp2 error is not amplified by a large |c|.
_VALUE_CHECK_MAX_ABS_EXPONENT = 8.0

# Relative tolerance for the value check. The 16-bit paths round the result to bf16 (2**-8
# relative) on top of the 21f approximation; the fp32 path is a few ULP mid-range, growing
# with |c| (see the kernel's comments), so 2**-14 leaves room without hiding a wrong sign,
# exponent or parity, which is what this module is for.
_RTOL_16BIT = 2.0**-5
_RTOL_FP32 = 2.0**-14

# Results beyond these magnitudes are the unchanged over/underflow tails and are not asserted.
_HUGE = 2.0**100
_TINY = 2.0**-100

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _uses_16bit_dst(formats: InputOutputFormat, dest_acc: DestAccumulation) -> bool:
    return (
        formats.input_format == DataFormat.Float16_b or dest_acc == DestAccumulation.No
    )


def _all_bf16_patterns() -> torch.Tensor:
    """Every bf16 bit pattern once, as exactly representable fp32 values."""
    patterns = (
        np.arange(LANES, dtype=np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    )
    return torch.from_numpy(patterns.copy())


def _fp32_bases() -> torch.Tensor:
    """65536 fp32 bases: specials, zeros, denormals, +-1 neighbourhoods, small integers, log-uniform."""
    f32 = np.finfo(np.float32)
    specials = [
        0.0,
        -0.0,
        INF,
        -INF,
        NAN,
        -NAN,
        f32.smallest_subnormal,
        -f32.smallest_subnormal,
        f32.smallest_normal,
        -f32.smallest_normal,
        f32.max,
        -f32.max,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        10.0,
        -10.0,
        0.1,
        -0.1,
    ]
    ints = np.arange(-32, 33, dtype=np.float32)
    # Dense neighbourhoods of +-1: 1 + k * 2**-20 for k in [-4096, 4096).
    k = np.arange(-4096, 4096, dtype=np.float64)
    near_one = (1.0 + k * 2.0**-20).astype(np.float32)
    near_one = np.concatenate([near_one, -near_one])
    fixed = np.concatenate([np.array(specials, dtype=np.float32), ints, near_one])

    gen = torch.Generator().manual_seed(0x5F3759DF)
    n_random = LANES - fixed.size
    # Log-uniform magnitude over the whole normal range, random sign, random mantissa.
    exponents = torch.empty(n_random, dtype=torch.float64).uniform_(
        -126.0, 128.0, generator=gen
    )
    magnitude = torch.pow(2.0, exponents)
    sign = torch.where(torch.rand(n_random, generator=gen) < 0.5, -1.0, 1.0).to(
        torch.float64
    )
    randoms = (sign * magnitude).to(torch.float32).numpy()
    randoms = np.where(
        np.isfinite(randoms), randoms, np.float32(f32.max)
    )  # 2**127 * 1.x may round to inf
    bases = np.concatenate([fixed, randoms])
    assert bases.size == LANES
    return torch.from_numpy(bases.astype(np.float32))


def _bases(formats: InputOutputFormat, dest_acc: DestAccumulation) -> torch.Tensor:
    if _uses_16bit_dst(formats, dest_acc):
        return _all_bf16_patterns()
    return _fp32_bases()


def _kernel_golden(x: torch.Tensor, c: float) -> torch.Tensor:
    """torch.pow in float64 with the kernel's documented departures from IEEE applied."""
    xd = x.to(torch.float64)
    g = torch.pow(xd, c)
    zero_base = xd == 0.0
    negative_base = xd < 0.0
    if c != c or c < 0.0:
        # 0**c for c < 0 (incl. -Inf) and for NaN c is NaN in the kernel (IEEE gives +Inf for c < 0).
        g = torch.where(zero_base, torch.full_like(g, NAN), g)
    if c != c or abs(c) == INF:
        # Non-finite exponents classify as non-integer: a negative base gives NaN.
        g = torch.where(negative_base, torch.full_like(g, NAN), g)
    return g


def _assertable(x: torch.Tensor, c: float) -> torch.Tensor:
    """Lanes whose result the kernel contract defines."""
    xd = x.to(torch.float64)
    finite_c = c == c and abs(c) != INF
    ok = torch.ones_like(xd, dtype=torch.bool)
    if c == 0.0:
        return ok  # x**0 == 1 for every x, including Inf and NaN
    ok &= torch.isfinite(
        xd
    )  # Inf / NaN bases: log2 of a non-finite is not defined by the kernel
    if not finite_c:
        ok &= (
            xd <= 0.0
        )  # positive bases with a non-finite exponent: 2**(+-Inf * log2 x) is not specified
    # Denormal bases: the pipeline may flush them (FTZ) before the kernel sees them.
    denormal = (xd != 0.0) & (xd.abs() < np.finfo(np.float32).smallest_normal)
    ok &= ~denormal
    return ok


def _check(
    device: torch.Tensor,
    golden: torch.Tensor,
    x: torch.Tensor,
    c: float,
    rtol: float,
    nan_as_inf: bool,
):
    """Return a list of (kind, count, example) mismatch descriptions; empty means pass."""
    d = device.to(torch.float64)
    g = golden
    xd = x.to(torch.float64)
    problems = []

    def report(kind, mask):
        n = int(mask.sum())
        if n:
            i = int(torch.nonzero(mask)[0])
            problems.append(
                (
                    kind,
                    n,
                    f"x={x[i].item()!r} -> got {d[i].item()!r}, want {g[i].item()!r}",
                )
            )

    g_nan = g != g
    d_nan = d != d
    nan_delivered = d_nan | (d == INF) if nan_as_inf else d_nan
    report("NaN expected, got a number", g_nan & ~nan_delivered)

    # Exact results the contract pins for every c: x**0 == 1, 0**c == 0 for c > 0.
    exact = ~g_nan & ((xd == 0.0) | (c == 0.0))
    report("exact result expected", exact & (d != g) & ~((g == 0.0) & (d == 0.0)))

    # Normal-range results: sign always, value for a moderate |c|. The tails are not asserted.
    normal = ~g_nan & ~exact & (g.abs() > _TINY) & (g.abs() < _HUGE)
    report("number expected, got NaN", normal & d_nan)
    normal &= ~d_nan
    report("sign", normal & (torch.sign(d) != torch.sign(g)))
    if abs(c) <= _VALUE_CHECK_MAX_ABS_EXPONENT:
        rel = (d - g).abs() / g.abs()
        report(f"relative error > {rtol:g}", normal & (rel > rtol))
    return problems


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    exponent_name=list(EXPONENTS),
)
def test_sfpu_unary_power_exponent(formats, dest_acc, exponent_name):
    c = EXPONENTS[exponent_name]
    if (
        exponent_name in _SATURATION_RANGE_EXPONENTS
        and TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
    ):
        pytest.skip(
            "Wormhole's kernel still derives parity from a saturating vSMag16 conversion: "
            "|c| > 32767 reads as non-integer and a negative base gives NaN"
        )

    x = _bases(formats, dest_acc)
    src_B = torch.zeros(TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1], dtype=torch.float32)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        INPUT_DIMENSIONS,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    tile_count = LANES // (TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1])

    configuration = TestConfig(
        "sources/sfpu_unary_power_exponent_test.cpp",
        formats,
        templates=[
            generate_input_dim(INPUT_DIMENSIONS, INPUT_DIMENSIONS),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(c)),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            x,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=1,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == LANES, "result length differs from the base tile"
    device = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format]).to(
        torch.float32
    )

    golden = _kernel_golden(x, c)
    assertable = _assertable(x, c)
    rtol = _RTOL_16BIT if _uses_16bit_dst(formats, dest_acc) else _RTOL_FP32

    problems = _check(
        device[assertable],
        golden[assertable],
        x[assertable],
        c,
        rtol,
        nan_as_inf=_uses_16bit_dst(formats, dest_acc),
    )
    if problems:
        detail = "\n".join(
            f"  {kind}: {n} lanes, e.g. {example}" for kind, n, example in problems
        )
        raise AssertionError(
            f"x**{c!r} ({formats.input_format.name}->{formats.output_format.name}, "
            f"dest_acc={dest_acc.name}) disagrees with the kernel contract on "
            f"{int(assertable.sum())} asserted lanes:\n{detail}"
        )
