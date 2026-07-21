# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import itertools
from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    BinarySFPUGolden,
    BroadcastGolden,
    get_golden_generator,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import DestAccumulation, DestSync, MathOperation, format_dict
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import DistributionKind, StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BROADCAST_TYPE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


def _skip_fp32_no_dest_acc(formats, dest_acc):
    """32-bit (Float32) inputs need a 32-bit dest, i.e. dest_acc=Yes."""
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")


def _skip_bh_float16_no_dest_acc(formats, dest_acc):
    """Blackhole can't run Float16 SFPU input without a 32-bit dest intermediate."""
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    bcast_dim=[
        LlkBroadcastType.None_,
        LlkBroadcastType.Row,
        LlkBroadcastType.Column,
        LlkBroadcastType.Scalar,
    ],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        # Eq/Ne moved to test_sfpu_binary_eq_ne: independent random draws are never
        # equal here, so the golden collapses to a constant — they need crafted paired
        # stimuli to exercise the equal branch.
        # Disabled: failing due to very small differences in generated stimuli
        # MathOperation.SfpuElwLt,
        # MathOperation.SfpuElwGt,
        # MathOperation.SfpuElwLe,
        # MathOperation.SfpuElwGe,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_float(
    formats,
    dest_acc,
    mathop,
    bcast_dim,
):
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # POW/XLOGY are only covered on the float formats: under Bfp8_b the coarse
    # quantization pushes small operands to values that produce -inf/NaN (log/pow),
    # so Bfp8_b coverage for these ops is intentionally skipped.
    if formats.input_format == DataFormat.Bfp8_b and mathop in (
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
    ):
        pytest.skip("Bfp8_b is not supported for POW/XLOGY coverage")

    if bcast_dim == LlkBroadcastType.Row and (
        dest_acc == DestAccumulation.Yes
        or is_format_combination_outlier(
            formats.input_format, formats.output_format, dest_acc
        )
    ):
        pytest.skip(
            "Row broadcast with FP32 dest: B2D datacopy uses MOVB2D which can't handle FP32 dest format conversion"
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=bcast_dim,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_div(formats, dest_acc):
    # DIV routes through the dedicated production kernel (calculate_sfpu_binary_div);
    # split out from the float sweep since the reciprocal path is precision-sensitive.
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        MathOperation.SfpuElwdiv,
        broadcast_type=LlkBroadcastType.None_,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    ],
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int(
    formats,
    dest_acc,
    mathop,
):
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
    )


# Deterministic edge-case coverage for the integer shift ops: shift amounts outside
# [0, 31] -> 0, arithmetic right-shift sign-extends, negatives shift correctly. INT32_MIN
# is excluded (sign-magnitude Dst can't represent -2^31); see the xfail test below and
# SFPU_INT32_SHIFT.md.

_INT32_MIN = -(2**31)

_SHIFT_EDGE_OPS = [
    MathOperation.SfpuElwRightShift,
    MathOperation.SfpuElwLeftShift,
    MathOperation.SfpuElwLogicalRightShift,
]

# Representative Int32 values: zero, small magnitudes of both signs, byte / halfword
# boundaries, the sign bit, an alternating bit pattern and the int32 extremes.
_SHIFT_EDGE_VALUES = [
    0,
    1,
    -1,
    2,
    -2,
    7,
    -8,
    255,
    -256,
    0x0000FFFF,
    -0x00010000,
    0x40000000,
    -0x40000000,
    0x55555555,
    -0x55555555,
    0x7FFFFFFF,  # INT32_MAX
    -0x80000000,  # INT32_MIN (filtered out per-op; see _build_shift_edge_case_src)
]

# Shift amounts spanning in-range values (0..31), the first out-of-range value (32),
# larger out-of-range values, and negative amounts. Everything outside [0, 31] must
# yield 0 to match the kernel.
_SHIFT_EDGE_AMOUNTS = [
    0,
    1,
    2,
    7,
    15,
    16,
    30,
    31,  # in-range
    32,
    33,
    40,
    63,
    100,
    1000,  # >= 32 -> 0
    -1,
    -5,
    -32,
    -1000,  # < 0 -> 0
]


def _shift_reference(mathop, value, shift):
    """Bit-exact reference for one (value, shift) pair: shifts outside [0, 31] -> 0, right
    shift arithmetic, logical right shift unsigned, left shift plain. Mirrors BinarySFPUGolden.
    """
    shift = int(shift)
    if shift < 0 or shift >= 32:
        return 0
    v = torch.tensor(int(value), dtype=torch.int32)
    if mathop == MathOperation.SfpuElwRightShift:
        return int(torch.bitwise_right_shift(v, shift))
    if mathop == MathOperation.SfpuElwLeftShift:
        return int(torch.bitwise_left_shift(v, shift))
    if mathop == MathOperation.SfpuElwLogicalRightShift:
        r = (int(value) & 0xFFFFFFFF) >> shift
        return r - 0x100000000 if r >= 0x80000000 else r
    raise ValueError(f"Unsupported shift op: {mathop}")


def _build_shift_edge_case_src(mathop):
    """Build a deterministic [64, 32] Int32 operand: tile 0 holds values, tile 1 holds
    per-element shift amounts (tilize pairs them by index). Walks the cartesian product of
    interesting (value, shift) pairs; pairs touching INT32_MIN are dropped (sign-magnitude
    Dst can't represent -2^31)."""
    pairs = [
        (v, s)
        for v, s in itertools.product(_SHIFT_EDGE_VALUES, _SHIFT_EDGE_AMOUNTS)
        if v != _INT32_MIN and _shift_reference(mathop, v, s) != _INT32_MIN
    ]
    num_elements = 32 * 32
    value_grid = [pairs[i % len(pairs)][0] for i in range(num_elements)]
    shift_grid = [pairs[i % len(pairs)][1] for i in range(num_elements)]
    return torch.tensor(value_grid + shift_grid, dtype=torch.int32)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=_SHIFT_EDGE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int_shift_edge_cases(
    formats,
    dest_acc,
    mathop,
):
    if TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        pytest.skip(
            reason="Blackhole shift kernels (left / arithmetic right / logical right) are "
            "unmigrated TTI microcode whose predicated out-of-range/sign handling breaks "
            "under INT32_2S_COMP for negative operands, so all three diverge from the "
            "two's-complement golden. See SFPU_INT32_SHIFT.md."
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_shift_edge_case_src(mathop),
    )


@pytest.mark.xfail(
    reason="Dst stores int32 as sign-magnitude with range +-(2^31 - 1). INT32_MIN "
    "(0x80000000) is 'negative zero' and cannot round-trip through Dst, so shifts that "
    "consume or produce it diverge from the two's-complement golden. This is a hardware "
    "limitation of the Wormhole SFPU load/store path; see SFPU_INT32_SHIFT.md.",
    strict=False,
)
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=_SHIFT_EDGE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int_shift_int32_min_unsupported(
    formats,
    dest_acc,
    mathop,
):
    # Every value lane is INT32_MIN shifted by 0: the golden expects INT32_MIN back, but
    # HW loads it as sign-magnitude "negative zero", so this is expected to fail.
    num_elements = 32 * 32
    value_grid = [_INT32_MIN] * num_elements
    shift_grid = [0] * num_elements
    src = torch.tensor(value_grid + shift_grid, dtype=torch.int32)
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=src,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    mathop=[
        MathOperation.SfpuBinaryMax,
        MathOperation.SfpuBinaryMin,
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_float_extended(formats, dest_acc, mathop):
    # max/min (SFPSWAP) and fmod/remainder (fp32 reciprocal) binary kernels with no
    # dedicated production BinaryOp; driven through the same in-DST harness as add/sub.
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # fmod/remainder divide by b via a reciprocal; Bfp8_b's coarse quantization blows up
    # the quotient for small divisors (mirrors the pow/xlogy Bfp8_b skip above).
    if formats.input_format == DataFormat.Bfp8_b and mathop in (
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
    ):
        pytest.skip("Bfp8_b is not supported for fmod/remainder coverage")

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=LlkBroadcastType.None_,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[
        MathOperation.SfpuBitwiseAnd,
        MathOperation.SfpuBitwiseOr,
        MathOperation.SfpuBitwiseXor,
    ],
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_bitwise(formats, dest_acc, mathop):
    # int32 bitwise AND/OR/XOR: exact on the full default int range.
    sfpu_binary(formats, dest_acc, mathop)


# Ops whose kernel interprets DST as unsigned; run them under UInt32 (the rest are Int32).
_UINT32_BINARY_OPS = {
    MathOperation.SfpuMaxUint32,
    MathOperation.SfpuMinUint32,
    MathOperation.SfpuRemainderUint32,
}

# int/uint binary ops sharing the same driver: dest_acc=Yes, single-format, and a per-op
# uniform positive stimuli range. Ranges keep operands (and results) non-negative and small
# enough to round-trip the sign-magnitude Dst packer plus any int->fp32 reciprocal the
# kernel uses. mathop -> (low, high).
_INT_BINARY_STIMULI = {
    # trunc/floor division < 2**24: exact int->fp32 reciprocal, trunc == floor, and the
    # sign-magnitude pack path can't round-trip the negatives these kernels would emit.
    MathOperation.SfpuDivInt32: (1.0, 8_000_000.0),
    MathOperation.SfpuDivInt32Floor: (1.0, 8_000_000.0),
    # binary-GCD on raw int32 bits (exact): strictly positive within the 31-bit budget.
    MathOperation.SfpuGcd: (1.0, 100_000.0),
    # lcm abs()es both operands and assumes |a|, |b| < 2**15.
    MathOperation.SfpuLcm: (1.0, 20_000.0),
    # int32 multiply low-32: operands < ~46340 so the product stays < 2**31 (non-negative).
    MathOperation.SfpuMulInt32: (1.0, 40_000.0),
    # int32/uint32 max/min via SFPSWAP: non-negative so signed/unsigned agree and round-trip.
    MathOperation.SfpuMaxInt32: (0.0, 1_000_000.0),
    MathOperation.SfpuMinInt32: (0.0, 1_000_000.0),
    MathOperation.SfpuMaxUint32: (0.0, 1_000_000.0),
    MathOperation.SfpuMinUint32: (0.0, 1_000_000.0),
    # remainder/fmod: non-negative operands, divisor >= 1 so every convention agrees;
    # kept < 2**24 for the exact int->fp32 reciprocal the quotient uses.
    MathOperation.SfpuRemainderInt32: (1.0, 10_000.0),
    MathOperation.SfpuFmodInt32: (1.0, 10_000.0),
    MathOperation.SfpuRemainderUint32: (1.0, 10_000.0),
}


@parametrize(
    mathop=list(_INT_BINARY_STIMULI),
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int_uniform(mathop, dest_acc):
    int_format = DataFormat.UInt32 if mathop in _UINT32_BINARY_OPS else DataFormat.Int32
    formats = InputOutputFormat(int_format, int_format)
    low, high = _INT_BINARY_STIMULI[mathop]
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=low, high=high),
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuRsubInt32],
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_rsub_int32(formats, dest_acc, mathop):
    sfpu_binary(formats, dest_acc, mathop, twos_complement=True)


# Number of faces per tile for the [64, 32] two-tile binary harness layout
# (a 32x32 tile is 4 faces of 16x16, and input_dimensions=[64, 32] is 8 faces).
_FACES_PER_TILE = 4


def _paired_two_tile_spec(a_face, b_face):
    """Fill operand tile0 (in0) and tile1 (in1) from *different* per-position data.

    The harness applies one per-face distribution to every face, so a plain callable makes
    in0 == in1. Keep `a_face` for the tile0 faces (0..3) and override the tile1 faces (4..7)
    with `b_face`, so position p pairs as (a_face[p], b_face[p]); tilize preserves it.
    """
    return StimuliSpec(
        distribution=a_face,
        seed=0,
        face_specs=[None] * _FACES_PER_TILE
        + [StimuliSpec(distribution=b_face, seed=0)] * _FACES_PER_TILE,
    )


def _mask_stimuli_spec():
    # mask zeroes data (in0) where mask (in1) is 0. Data and mask are separate tiles: keep
    # data strictly non-zero (1..8) and zero ~1/3 of the mask, so a passthrough kernel fails.
    def data_face(size, dtype, generator):
        j = torch.arange(size, dtype=torch.float32)
        return (1.0 + (j % 8)).to(dtype)  # 1..8, always non-zero

    def mask_face(size, dtype, generator):
        j = torch.arange(size, dtype=torch.float32)
        return torch.where(j % 3 == 0, 0.0, 1.0).to(dtype)  # ~1/3 exact zeros

    return _paired_two_tile_spec(data_face, mask_face)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuMask],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_mask(formats, dest_acc, mathop):
    # float mask: data at tile0, mask at tile1. Output is data where mask != 0, else 0.
    # Crafted stimuli so the mask carries real zeros.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=LlkBroadcastType.None_,
        spec_A=_mask_stimuli_spec(),
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuAtan2],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_atan2(formats, dest_acc, mathop):
    # atan2(y, x): y = tile0, x = tile1. Signed [-5, 5] gives mixed signs so all quadrants
    # (and the |y|>=|x| / x<0 branches) are exercised; minimax approximation matched under PCC.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0),
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    mathop=[MathOperation.SfpuEqInt, MathOperation.SfpuNeInt],
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_eq_ne_int(formats, dest_acc, mathop):
    # int32 eq/ne via calculate_binary_eq_int (exact 0/1 over the raw INT32 dest bits).
    # Reuse the paired eq/ne stimuli so ~50% of positions compare equal — the equal branch
    # a plain random int sweep would essentially never hit.
    sfpu_binary(formats, dest_acc, mathop, spec_A=_eq_ne_stimuli_spec())


def _isclose_stimuli_spec():
    # isclose is a predicate on paired operands (a = tile0, b = tile1). Fill the two tiles
    # from different data so even p -> identical (isclose 1), odd p -> differ by 2.0
    # (isclose 0); the 2.0 gap dwarfs the tolerance so the decision is unambiguous.
    def a_face(size, dtype, generator):
        j = torch.arange(size, dtype=torch.float32)
        return (1.0 + (j % 8)).to(dtype)  # 1..8, strictly positive

    def b_face(size, dtype, generator):
        j = torch.arange(size, dtype=torch.float32)
        base = 1.0 + (j % 8)
        return (base + torch.where(j % 2 == 0, 0.0, 2.0)).to(dtype)

    return _paired_two_tile_spec(a_face, b_face)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuIsclose],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_isclose(formats, dest_acc, mathop):
    # isclose(a, b) = |a - b| <= atol + rtol*|b|, a = tile0, b = tile1. torch default
    # tolerances (fixed in the C++ dispatch); crafted stimuli give a non-constant 0/1 mix.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=_isclose_stimuli_spec(),
    )


def _eq_ne_stimuli_spec():
    # Eq/Ne compare paired operands (a = tile0, b = tile1). Fill the two tiles so even p ->
    # identical (Eq 1), odd p -> differ by 1.0 (Eq 0), a clean ~50/50 mix.
    def a_face(size, dtype, generator):
        j = torch.arange(size, dtype=torch.float32)
        return (1.0 + (j % 8)).to(dtype)  # 1..8

    def b_face(size, dtype, generator):
        j = torch.arange(size, dtype=torch.float32)
        base = 1.0 + (j % 8)
        return (base + torch.where(j % 2 == 0, 0.0, 1.0)).to(dtype)

    return _paired_two_tile_spec(a_face, b_face)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuElwEq, MathOperation.SfpuElwNe],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_eq_ne(formats, dest_acc, mathop):
    # Eq/Ne(a, b) with a = tile0, b = tile1. Crafted paired stimuli give a non-constant 0/1
    # golden so the equal branch is exercised (the default random sweep never is).
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=_eq_ne_stimuli_spec(),
    )


def _logsigmoid_stimuli_spec():
    # logsigmoid(x) = -softplus(-x). in1 (exp(-x)) is only read in the x > 4 branch, so
    # restrict x to [-8, 3.9] (never uses in1) and sweep the passthrough (x < -4) and
    # polynomial (-4 < x < 4) branches. The distribution is invoked per 16x16 face (size 256).
    def dist(size, dtype, generator):
        return torch.linspace(-8.0, 3.9, size).to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=[MathOperation.SfpuLogsigmoid],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_logsigmoid(formats, dest_acc, mathop):
    # logsigmoid(x) with x = tile0. Piecewise poly/passthrough approximation matched under
    # PCC; x swept over [-8, 3.9]. The x > 4 (-exp(-x)) branch needs a device-computed
    # exp(-x) operand the shared harness can't provide, left to a future driver.
    _skip_fp32_no_dest_acc(formats, dest_acc)

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        spec_A=_logsigmoid_stimuli_spec(),
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    mathop=[MathOperation.SfpuAddTopRow],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_add_top_row(formats, dest_acc, mathop):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip(
            "32-bit integer formats require DestAccumulation.Yes (HW cannot unpack into SrcA/SrcB)"
        )

    input_dimensions = [64, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        0,
        1,
        0,
        1,
        input_dimensions,
        formats.output_format,
    )

    golden_tensor = (
        golden_tensor.view([32, 32])
        if golden_tensor.shape == torch.Size([1024])
        else golden_tensor.view(input_dimensions)
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
            BROADCAST_TYPE(LlkBroadcastType.None_),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).view(input_dimensions)

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


def sfpu_binary(
    formats,
    dest_acc,
    mathop,
    broadcast_type=None,
    src_A_override=None,
    spec_A=None,
    spec_B=None,
    twos_complement=False,
):

    # FP32 destination tiles occupy twice the register space. Keep four full destination
    # blocks for those formats and four blocks of eight tiles for the remaining formats.
    input_dimensions = [128, 128] if formats.input_format.is_32_bit() else [256, 128]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
        spec_B=spec_B,
    )

    # The kernel only consumes buffer_A (operand 0 = tile 0, operand 1 = tile 1), so an
    # explicit src_A fully controls inputs for edge cases; src_B stays random but unused.
    if src_A_override is not None:
        override = src_A_override.to(src_A.dtype).flatten()
        if src_A.numel() % override.numel() != 0:
            raise ValueError(
                "SFPU binary override must contain a whole number of tile pairs"
            )
        src_A = override.repeat(src_A.numel() // override.numel())

    golden_src = src_A
    if broadcast_type is not None and broadcast_type != LlkBroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src = generate_broadcast_golden(
            broadcast_type,
            src_A,
            (
                formats.input_format
                if formats.input_format != DataFormat.Bfp8_b
                else DataFormat.Float16_b
            ),
            tile_cnt=tile_cnt_A,
        )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_format = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Bfp8_b
        else formats.input_format
    )
    elements_per_pair = 2 * 32 * 32
    golden_tensor = torch.cat(
        [
            generate_golden(
                mathop,
                golden_src[offset : offset + elements_per_pair],
                0,
                1,
                0,
                32,
                [64, 32],
                golden_format,
            ).flatten()
            for offset in range(0, golden_src.numel(), elements_per_pair)
        ]
    )

    # ONLY Blackhole needs this for some reason
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    bcast = broadcast_type if broadcast_type else LlkBroadcastType.None_

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
            BROADCAST_TYPE(bcast),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            twos_complement=twos_complement,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# SFPU binary with row/column broadcast (BCAST_COL / BCAST_ROW). Uses its own 3-tile
# kernel source (sources/sfpu_binary_bcast_test.cpp) with a custom init and full-tile
# driver; InstrModLoadStore::DEFAULT works for any float dest format (compute is FP32).


class BroadcastType(Enum):
    # Values must match ckernel::BroadcastType in llk_defs.h
    # (NONE=0, COL=1, ROW=2, SCALAR=3) because the kernel does
    # `static_cast<BroadcastType>(BCAST_DIM_VAL)`.
    COL = 1
    ROW = 2


@dataclass
class SFPU_BCAST_DIM(TemplateParameter):
    bcast_dim: BroadcastType

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t BCAST_DIM_VAL = {self.bcast_dim.value};"


@dataclass
class INPUT_TILE_A(TemplateParameter):
    """Base DST tile index for input A.

    The kernel derives the other tile indices from this single value:
      INPUT_TILE_A      -> data tile
      INPUT_TILE_A + 1  -> bcast tile
      INPUT_TILE_A + 2  -> result tile
    """

    tile_index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t INPUT_TILE_A_VAL = {self.tile_index};"


_BCAST_BINARY_OPS = {
    MathOperation.SfpuElwadd: torch.add,
    MathOperation.SfpuElwsub: torch.sub,
    MathOperation.SfpuElwmul: torch.mul,
}


def _golden_sfpu_binary_bcast(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    bcast_dim: BroadcastType,
    op,
    stimuli_format: DataFormat,
) -> torch.Tensor:
    """Golden for the SFPU bcast kernel (single 32x32 tile): broadcast in row-major space,
    then tilize to the packer's layout. `stimuli_format` drives tilize precision (Float16_b
    for Bfp8_b inputs, since the unpacker converts Bfp8_b -> Float16_b in dest)."""
    a = src_A.flatten()[:1024].reshape(32, 32)
    b = src_B.flatten()[:1024].reshape(32, 32)

    if bcast_dim == BroadcastType.ROW:
        b_bcast = b[0].unsqueeze(0).expand_as(b)
    else:
        b_bcast = b[:, 0].unsqueeze(1).expand_as(b)

    golden_rm = op(a, b_bcast.contiguous()).flatten()
    return tilize(golden_rm, stimuli_format=stimuli_format)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    bcast_dim=[BroadcastType.ROW, BroadcastType.COL],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_bcast(
    formats,
    bcast_dim,
    mathop,
    dest_acc,
):
    _skip_fp32_no_dest_acc(formats, dest_acc)
    _skip_bh_float16_no_dest_acc(formats, dest_acc)

    # Mirror sfpu_binary(): on Blackhole, Float16/Float32 inputs require
    # dest_acc=Yes (32-bit dest), so silently upgrade the parametrized value.
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Bfp8_b stimuli are effectively Float16_b in dest after unpack; golden
    # computes and tilizes at that precision to match.
    golden_format = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Bfp8_b
        else formats.input_format
    )
    golden_tensor = _golden_sfpu_binary_bcast(
        src_A, src_B, bcast_dim, _BCAST_BINARY_OPS[mathop], golden_format
    )

    # Only FP32 inputs with dest_acc=Yes take the unpack-to-dest path; all
    # other float formats go through srcA + MATH datacopy into dest.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/sfpu_binary_bcast_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            SFPU_BCAST_DIM(bcast_dim),
            INPUT_TILE_A(tile_index=0),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilize(src_A, stimuli_format=formats.input_format),
            formats.input_format,
            tilize(src_B, stimuli_format=formats.input_format),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result ({len(res_from_L1)}) and golden ({len(golden_tensor)}) size mismatch"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
