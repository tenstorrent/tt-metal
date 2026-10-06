# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK SFPU quant-family tests (quant / requant / dequant) for Blackhole.

Scope: the eleven compute-API entry points of tt_metal/hw/inc/api/compute/quantization.h
(int32 / uint8 / int8 outputs, int32 / int8 inputs), as ttnn's binary_ng quantize ops emit
them. They have no ckernel::BinaryOp, so the kernels are reached through the dedicated
sources/sfpu_quant_test.cpp, which mirrors the production wrappers one to one and is selected
by SFPU_QUANT_VARIANT.

Both operands travel as raw Int32 words (two's complement, unpacked straight to a 32-bit Dest)
and the result is read back as raw Int32 words, so an fp32 input or scale is delivered as its
bit pattern and every output word -- int32, uint8-in-a-word or int8 byte -- is compared
bit-for-bit against an exact Python emulation of the kernel. Stimuli are chosen so that the
fp32 multiply-add is exact and never lands on a rounding tie; a separate test drives the
special lanes (+-0, +-Inf, +-NaN, denormals, the -128 / 127 / 255 saturation edges and exact
ties) and asserts only the lanes whose result the kernel defines, logging the rest.
"""

from fractions import Fraction

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import ApproximationMode, DestAccumulation, DestSync
from helpers.logger import logger
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    QUANT_VARIANTS,
    SFPU_QUANT_VARIANT,
    TILE_COUNT,
    ZERO_POINT,
    generate_input_dim,
)

ELEMENTS_PER_TILE = 1024

# Variants by the width of operand 0 (the quantity being (re/de)quantized) and of the result.
_FP32_IN = {"QUANT", "QUANT_UINT8", "QUANT_INT8"}
_INT8_IN = {
    "REQUANT_INT8_IN",
    "REQUANT_INT8_IN_UINT8_OUT",
    "REQUANT_INT8_IN_INT8_OUT",
    "DEQUANT_INT8",
}
_INT8_OUT = {"QUANT_INT8", "REQUANT_INT8", "REQUANT_INT8_IN_INT8_OUT"}
_UINT8_OUT = {"QUANT_UINT8", "REQUANT_UINT8", "REQUANT_INT8_IN_UINT8_OUT"}
_FP32_OUT = {"DEQUANT", "DEQUANT_INT8"}


def _f32_bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.int32)


def _bits_f32(words) -> np.ndarray:
    """fp32 values from 32-bit patterns given as int32 or as unsigned Python ints."""
    return (
        (np.asarray(words, dtype=np.int64) & 0xFFFFFFFF)
        .astype(np.uint32)
        .view(np.float32)
    )


_F32_MAX = Fraction(float(np.finfo(np.float32).max))


def _exact_f32(fr: Fraction):
    """The fp32 equal to *fr*; +-Inf when |fr| overflows fp32; None when *fr* is finite in fp32
    but not exactly representable.

    Only exact conversions are accepted, so the golden never has to model the SFPU's fp32
    rounding: the stimuli below are built so that every multiply-add result is exact.
    """
    if abs(fr) > _F32_MAX:
        return np.float32(np.inf) if fr > 0 else np.float32(-np.inf)
    d = float(fr)
    if Fraction(d) != fr:
        return None
    f = np.float32(d)
    if Fraction(float(f)) != fr:
        return None
    return f


def _saturate_infinite(
    variant: str, v_inf: float, expected: np.ndarray, defined: np.ndarray, i: int
):
    """Result of an infinite value on each output path (sign decides the saturation side)."""
    if variant in _FP32_OUT:
        expected[i] = _f32_bits([v_inf])[0]
    elif variant in _INT8_OUT:
        expected[i] = 0x7F if v_inf > 0 else 0x80
    elif variant in _UINT8_OUT:
        if v_inf < 0:
            defined[i] = False  # magnitude of -Inf on the unclamped uint8 path
        else:
            expected[i] = 255
    else:
        expected[i] = 127 if v_inf > 0 else -127


def _round_half_away(fr: Fraction) -> tuple[int, bool]:
    """(round-half-away-from-zero of fr, whether fr was an exact tie)."""
    tie = (fr - (fr.numerator // fr.denominator)) == Fraction(1, 2)
    if fr >= 0:
        n = (fr + Fraction(1, 2)).numerator // (fr + Fraction(1, 2)).denominator
    else:
        m = -fr + Fraction(1, 2)
        n = -(m.numerator // m.denominator)
    return n, tie


def _requant_input_as_f32(variant: str, word: int) -> Fraction:
    """Operand 0 as the kernel's SFPCAST sees it: int32 two's complement, or the UInt8-unpacked
    int8 byte after the kernel's `^ 0x80` unbias (excess-128)."""
    if variant in _INT8_IN:
        return Fraction((word & 0xFF) ^ 0x80)
    return Fraction(int(np.int32(word)))


def quant_golden(
    variant: str, in0_words: np.ndarray, scale_words: np.ndarray, zero_point_bits: int
):
    """Bit-exact emulation of one quant-family kernel on raw Int32 words.

    Returns (expected int32 words, defined mask, tie mask). A lane is *defined* when the
    kernel's result follows from the documented semantics without relying on an unspecified
    hardware choice (NaN inputs, a rounding tie, an inexact fp32 multiply-add, or a negative
    value on the unclamped FP32_TO_UINT8 path, which returns the magnitude). Ties are reported
    separately so the caller can check the hardware's tie rule instead of assuming one.
    """
    in0_words = np.asarray(in0_words, dtype=np.int32)
    scale_words = np.asarray(scale_words, dtype=np.int32)
    n = in0_words.size
    expected = np.zeros(n, dtype=np.int32)
    defined = np.ones(n, dtype=bool)
    tie = np.zeros(n, dtype=bool)

    zp_f = _bits_f32([zero_point_bits])[0]
    scales = _bits_f32(scale_words)
    in0_f32 = _bits_f32(in0_words) if variant in _FP32_IN else None

    for i in range(n):
        s = scales[i]
        if not np.isfinite(s):
            defined[i] = False
            continue
        if variant in _FP32_IN:
            a = in0_f32[i]
            if not np.isfinite(a):
                # +-Inf saturate (int outputs) or propagate (fp32 outputs); NaN is unspecified.
                if np.isnan(a) or not np.isfinite(zp_f):
                    defined[i] = False
                    continue
                if s == 0:
                    defined[i] = False  # Inf * 0 = NaN
                    continue
                _saturate_infinite(variant, float(a * s), expected, defined, i)
                continue
            a_fr = Fraction(float(a))
        else:
            if variant not in _INT8_IN and int(in0_words[i]) == -(2**31):
                # INT32_MIN has no sign-magnitude encoding: the kernel's two's-complement ->
                # sign-magnitude SFPCAST+SFPSETSGN turns it into -0, which casts to 0.0 (measured
                # on silicon, pre-existing). Not a value ttnn's int8-range requant/dequant can see.
                defined[i] = False
                continue
            a_fr = _requant_input_as_f32(variant, int(in0_words[i]))

        if not np.isfinite(zp_f):
            defined[i] = False
            continue
        s_fr = Fraction(float(s))
        zp_fr = Fraction(float(zp_f))

        if variant in _FP32_OUT:
            # dequant: (A + LREG2) * B with LREG2 = the bits the caller passed (-zero_point).
            summ = _exact_f32(a_fr + zp_fr)
            if summ is None:
                defined[i] = False
                continue
            if np.isinf(summ):
                _saturate_infinite(
                    variant,
                    float(summ) * (1.0 if s > 0 else -1.0),
                    expected,
                    defined,
                    i,
                )
                continue
            prod = _exact_f32(Fraction(float(summ)) * s_fr)
            if prod is None:
                defined[i] = False
                continue
            expected[i] = _f32_bits([prod])[0]
            continue

        bias = Fraction(128) if variant in _INT8_OUT else Fraction(0)
        v_fr_exact = a_fr * s_fr + zp_fr + bias
        v = _exact_f32(v_fr_exact)
        if v is None:
            defined[i] = False
            continue
        if np.isinf(v):
            _saturate_infinite(variant, float(v), expected, defined, i)
            continue
        v_fr = Fraction(float(v))

        if variant in _INT8_OUT:
            # max(v, 0) -> FP32_TO_UINT8 (saturating [0, 255]) -> ^ 0x80
            clamped = v_fr if v_fr > 0 else Fraction(0)
            r, is_tie = _round_half_away(clamped)
            tie[i] = is_tie
            u = min(max(r, 0), 255)
            expected[i] = u ^ 0x80
        elif variant in _UINT8_OUT:
            if v_fr < 0:
                defined[i] = (
                    False  # unclamped FP32_TO_UINT8 of a negative: magnitude, unspecified here
                )
                continue
            r, is_tie = _round_half_away(v_fr)
            tie[i] = is_tie
            expected[i] = min(r, 255)
        else:
            # FP32_TO_INT8 is sign-magnitude with a 7-bit magnitude: saturates at +-127.
            r, is_tie = _round_half_away(v_fr)
            tie[i] = is_tie
            expected[i] = min(max(r, -127), 127)

    return expected, defined, tie


def _run_variant(
    variant: str, in0_words: np.ndarray, scale_words: np.ndarray, zero_point_bits: int
):
    """Run one quant-family variant on interleaved (in0, scale) tile pairs; returns the result words."""
    assert (
        in0_words.shape == scale_words.shape and in0_words.size % ELEMENTS_PER_TILE == 0
    )
    pairs = in0_words.size // ELEMENTS_PER_TILE
    tile_cnt = 2 * pairs
    input_dimensions = (
        [32 * tile_cnt // 4, 32 * 4] if tile_cnt % 4 == 0 else [32 * tile_cnt, 32]
    )

    # buffer_A holds both operands: even tile = operand 0, odd tile = scale.
    buffer = np.empty((pairs, 2, ELEMENTS_PER_TILE), dtype=np.int32)
    buffer[:, 0, :] = in0_words.reshape(pairs, ELEMENTS_PER_TILE)
    buffer[:, 1, :] = scale_words.reshape(pairs, ELEMENTS_PER_TILE)
    src_A = torch.from_numpy(buffer.reshape(-1).copy())
    src_B = torch.zeros_like(src_A)

    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = (
        DestAccumulation.Yes
    )  # 32-bit Dest: every quant kernel loads/stores 32-bit words
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/sfpu_quant_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            SFPU_QUANT_VARIANT(variant),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            ZERO_POINT(zero_point_bits),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            twos_complement=True,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        compile_time_formats=True,
    )
    res = configuration.run().result
    res = np.asarray(
        torch.as_tensor(res).to(torch.int32).flatten().numpy(), dtype=np.int32
    )
    assert res.size == tile_cnt * ELEMENTS_PER_TILE
    # Even tiles carry the results; odd tiles are the scale operand packed back unchanged.
    res = res.reshape(pairs, 2, ELEMENTS_PER_TILE)
    scale_back = res[:, 1, :].reshape(-1)
    assert np.array_equal(
        scale_back, scale_words
    ), "the untouched scale tiles did not round-trip through Dest bit-exactly"
    return res[:, 0, :].reshape(-1)


def _hex(words) -> list[str]:
    return [
        f"0x{int(np.uint32(np.int32(w))):08x}" for w in np.asarray(words).reshape(-1)
    ]


def _assert_defined_lanes(
    variant, in0_words, scale_words, result, expected, defined, label
):
    mismatch = defined & (result != expected)
    if mismatch.any():
        idx = np.flatnonzero(mismatch)[:16]
        rows = [
            f"  lane {i}: in0={_hex([in0_words[i]])[0]} scale={_hex([scale_words[i]])[0]} "
            f"got={_hex([result[i]])[0]} expected={_hex([expected[i]])[0]}"
            for i in idx
        ]
        raise AssertionError(
            f"{variant} {label}: {int(mismatch.sum())} of {int(defined.sum())} defined lanes differ\n"
            + "\n".join(rows)
        )


# ---------------------------------------------------------------------------------------------
# Main sweep: exact, tie-free stimuli covering the in-range and saturating regions of every variant.
# ---------------------------------------------------------------------------------------------

# Scales are powers of two in [1/8, 4] so an operand with at most 4 fractional bits keeps the
# multiply-add exact in fp32. Ties are avoided by construction: fp32 inputs are odd sixteenths,
# so a*s has an odd/128 .. odd/4 fractional part and an integer zero-point keeps it that way;
# integer inputs (requant) get a zero-point with a 1/16 fraction instead, so a*s (a multiple of
# 1/8) plus 1/16 is always odd/16. Neither is ever .5 or a whole number.
_SCALES = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
_ZERO_POINT_FP32_IN = 5.0
_ZERO_POINT_INT_IN = 5.0625
_SEED = 0


def _main_stimuli(variant: str, generator: np.random.Generator, n: int):
    """(in0 words, scale words, zero-point bits) for the main sweep of *variant*."""
    scales = generator.choice(_SCALES, size=n).astype(np.float32)
    if variant in _FP32_IN:
        # odd sixteenths in [-512, 512): covers both saturation sides for every scale
        k = generator.integers(-4096, 4096, size=n)
        in0 = ((2 * k + 1) / 16.0).astype(np.float32)
        if variant in _UINT8_OUT:
            in0 = np.abs(
                in0
            )  # the uint8 path has no clamp; keep v >= 0 (see quant_golden)
        in0_words = _f32_bits(in0)
    elif variant in _INT8_IN:
        # raw int8 bytes as the UInt8 unpacker delivers them: the low byte of a zero-extended word
        in0_words = generator.integers(0, 256, size=n).astype(np.int32)
    else:
        # int32 two's complement in [-2048, 2048): exact in fp32, saturates for scale >= 1/8
        in0_words = generator.integers(-2048, 2048, size=n).astype(np.int32)
        if variant in _UINT8_OUT:
            in0_words = np.abs(in0_words).astype(np.int32)
    zp = _ZERO_POINT_FP32_IN if variant in _FP32_IN else _ZERO_POINT_INT_IN
    if variant in _FP32_OUT:
        zp = -zp  # dequant takes the bits of -zero_point
    return in0_words, _f32_bits(scales), int(np.uint32(_f32_bits([zp])[0]))


@parametrize(
    variant=list(QUANT_VARIANTS),
    input_dimensions=[[128, 128]],  # 16 tiles = 8 (in0, scale) pairs of 1024 lanes each
)
def test_sfpu_quant(variant: str, input_dimensions: list[int]):
    n = (
        (input_dimensions[0] // 32)
        * (input_dimensions[1] // 32)
        // 2
        * ELEMENTS_PER_TILE
    )
    in0_words, scale_words, zp_bits = _main_stimuli(
        variant, np.random.default_rng(_SEED), n
    )
    expected, defined, tie = quant_golden(variant, in0_words, scale_words, zp_bits)
    assert defined.all(), "main-sweep stimuli must be fully defined (exact, tie-free)"
    assert not tie.any(), "main-sweep stimuli must not contain rounding ties"

    result = _run_variant(variant, in0_words, scale_words, zp_bits)
    _assert_defined_lanes(
        variant, in0_words, scale_words, result, expected, defined, "main sweep"
    )


# ---------------------------------------------------------------------------------------------
# Special lanes: the edges each kernel's clamp / saturation / conversion path is built around.
# ---------------------------------------------------------------------------------------------

_F32_SPECIALS = [
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    1.0e-45,
    -1.0e-45,  # smallest denormal, both signs
    1.17549435e-38,
    -1.17549435e-38,  # FLT_MIN
    3.4028235e38,
    -3.4028235e38,  # FLT_MAX
    1.0e6,
    -1.0e6,
    300.0,
    -300.0,
    256.0,
    255.5,
    255.0,
    254.5,
    129.0,
    128.5,
    128.0,
    127.5,
    127.25,
    127.0,
    126.5,
    126.75,
    -126.5,
    -126.75,
    -127.0,
    -127.25,
    -127.5,
    -127.75,
    -128.0,
    -128.25,
    -128.5,
    -128.75,
    -129.0,
    -200.0,
    200.0,
    0.25,
    -0.25,
    0.5,
    -0.5,
    0.75,
    -0.75,
    1.5,
    -1.5,
    2.5,
    -2.5,
    3.5,
    -3.5,
    1.0,
    -1.0,
]
_F32_SPECIAL_WORDS = [
    0x7FC00000,
    0xFFC00000,  # canonical quiet NaN, both signs
    0x7F800001,
    0xFF800001,  # signalling NaN payloads, both signs
    0x7FFFFFFF,
    0xFFFFFFFF,  # all-ones payload NaN, both signs
]
_INT32_SPECIALS = [
    0,
    1,
    -1,
    127,
    128,
    -127,
    -128,
    -129,
    255,
    256,
    -255,
    -256,
    1000,
    -1000,
    2**23,
    -(2**23),
    2**24,
    -(2**24),
    2**31 - 1,
    -(2**31),
    -(2**31) + 1,
]
_INT8_BYTE_SPECIALS = [0x00, 0x01, 0x7F, 0x80, 0x81, 0xFF, 0xFE, 0x40, 0xC0]


def _special_stimuli(variant: str):
    if variant in _FP32_IN:
        words = list(_f32_bits(_F32_SPECIALS)) + [
            int(np.int32(np.uint32(w))) for w in _F32_SPECIAL_WORDS
        ]
    elif variant in _INT8_IN:
        words = _INT8_BYTE_SPECIALS
    else:
        words = _INT32_SPECIALS
    words = np.array(words, dtype=np.int32)
    # every special against scale 1.0 and, for the fp32 inputs, also against 0.5 and 2.0
    scales = [1.0] if variant not in _FP32_IN else [1.0, 0.5, 2.0]
    in0 = np.concatenate([words for _ in scales])
    sc = np.concatenate([np.full(words.size, s, dtype=np.float32) for s in scales])
    pad = (-in0.size) % ELEMENTS_PER_TILE
    in0 = np.concatenate([in0, np.zeros(pad, dtype=np.int32)])
    sc = np.concatenate([sc, np.ones(pad, dtype=np.float32)])
    return in0, _f32_bits(sc), in0.size - pad


# Plain pytest.mark.parametrize: the harness's @parametrize hands a lone parameter over as a
# 1-tuple (see test_exponential_clamp_negative for the same choice).
@pytest.mark.parametrize("variant", list(QUANT_VARIANTS))
def test_sfpu_quant_special_lanes(variant: str):
    """Edge lanes with zero-point 0: asserts every lane the kernel defines, and reports the
    hardware's choice on the rest (NaN inputs, rounding ties, the unclamped uint8 path).
    """
    in0_words, scale_words, n_real = _special_stimuli(variant)
    zp_bits = 0
    expected, defined, tie = quant_golden(variant, in0_words, scale_words, zp_bits)
    result = _run_variant(variant, in0_words, scale_words, zp_bits)

    # Ties are asserted against ONE rule for the whole variant rather than assumed:
    # sfpi 7.83.0 documents Blackhole SFP_STOCH_RND mode 0 as round-half-away.
    tie_lanes = np.flatnonzero(tie[:n_real])
    if tie_lanes.size:
        half_away_ok = np.array_equal(result[tie_lanes], expected[tie_lanes])
        logger.info(
            f"{variant}: {tie_lanes.size} tie lanes follow round-half-away: {half_away_ok}; "
            + ", ".join(
                f"in0={_hex([in0_words[i]])[0]}*{_bits_f32([scale_words[i]])[0]:g} -> {_hex([result[i]])[0]}"
                for i in tie_lanes[:12]
            )
        )
        assert (
            half_away_ok
        ), f"{variant}: FP32_TO_INT8/UINT8 ties do not round half away from zero"

    undefined = np.flatnonzero(~defined[:n_real])
    if undefined.size:
        logger.info(
            f"{variant}: implementation-defined lanes -> "
            + ", ".join(
                f"in0={_hex([in0_words[i]])[0]}*{_bits_f32([scale_words[i]])[0]:g} -> {_hex([result[i]])[0]}"
                for i in undefined
            )
        )

    non_tie_defined = defined.copy()
    non_tie_defined[tie] = False
    non_tie_defined[n_real:] = False
    _assert_defined_lanes(
        variant,
        in0_words,
        scale_words,
        result,
        expected,
        non_tie_defined,
        "special lanes",
    )
