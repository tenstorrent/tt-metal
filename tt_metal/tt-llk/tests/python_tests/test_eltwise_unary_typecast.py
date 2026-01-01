# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK SFPU typecast tests.

Scope: the **full ttnn typecast matrix** — every directed dtype pair that
``ttnn.typecast`` exercises end-to-end. That spans float<->float, float<->int,
int<->int and all block-float (Bfp8_b / Bfp4_b) conversions. Same-dtype pairs
and the ``int32<->uint32`` pair (not a kernel pair) are excluded.
"""

import math
import struct

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import (
    BLACKHOLE_DATA_FORMAT_ENUM_VALUES,
    QUASAR_DATA_FORMAT_ENUM_VALUES,
    WORMHOLE_DATA_FORMAT_ENUM_VALUES,
    DataFormat,
    InputOutputFormat,
)
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    TypecastGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    MathOperation,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TYPECAST_FORMATS,
    DestSync,
    generate_input_dim,
)
from helpers.utils import passed_test, result_tensor

# The full ttnn typecast dtype set (analysis doc section 5).
_TYPECAST_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
    DataFormat.Int32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.UInt8,
]

# Pairs the kernel LUT does not implement (and ttnn does not test): int32<->uint32.
_EXCLUDED_PAIRS = {
    (DataFormat.Int32, DataFormat.UInt32),
    (DataFormat.UInt32, DataFormat.Int32),
}

# Formats the current architecture's data-format enum actually supports. Used to
# drop unsupported pairs at collection time so we never generate tests we would
# otherwise have to skip in the test body.
_ARCH_SUPPORTED_FORMATS = {
    ChipArchitecture.WORMHOLE: WORMHOLE_DATA_FORMAT_ENUM_VALUES,
    ChipArchitecture.BLACKHOLE: BLACKHOLE_DATA_FORMAT_ENUM_VALUES,
    ChipArchitecture.QUASAR: QUASAR_DATA_FORMAT_ENUM_VALUES,
}[get_chip_architecture()]

# Every directed pair ttnn.typecast exercises end-to-end, restricted to pairs
# whose input and output formats are both supported on the current architecture.
TYPECAST_PAIRS = [
    InputOutputFormat(in_fmt, out_fmt)
    for in_fmt in _TYPECAST_FORMATS
    for out_fmt in _TYPECAST_FORMATS
    if in_fmt != out_fmt
    and (in_fmt, out_fmt) not in _EXCLUDED_PAIRS
    and in_fmt in _ARCH_SUPPORTED_FORMATS
    and out_fmt in _ARCH_SUPPORTED_FORMATS
]


def _whole_number_float_spec(high: int) -> StimuliSpec:
    """Float stimuli restricted to whole numbers in ``[0, high)``.

    Whole numbers make float->int conversions exact regardless of whether the
    kernel truncates (int32/uint32) or rounds (uint16/uint8), and keep the
    values inside the range each format represents exactly so the golden is
    lossless. A small ``high`` is used when a block-float format is involved so
    the shared-exponent quantization is (near-)exact within each 16-elem block.
    """

    def dist(size, dtype, generator):
        return torch.randint(0, high, (size,), generator=generator).to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


def _preserve_fp32_precision(formats: InputOutputFormat) -> bool:
    """preserve_fp32_precision exactly as ttnn.typecast computes it.

    ttnn uses this single flag for two things: it forces fp32 dest acc (see
    _production_dest_acc) and it selects UnpackToDestFp32 in the program
    factories, so the test reuses it for both.
    """
    in_fmt = formats.input_format
    out_fmt = formats.output_format
    # bf16 / block-float inputs that promote to 32-bit Dest when packed to UInt8.
    bf_family = (DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b)
    return (
        in_fmt == DataFormat.Float32
        or (out_fmt == DataFormat.UInt8 and in_fmt in bf_family)
        or (in_fmt == DataFormat.UInt16 and out_fmt == DataFormat.UInt8)
        or (in_fmt == DataFormat.UInt8)
    )


def _production_dest_acc(formats: InputOutputFormat) -> list[DestAccumulation]:
    """Pick dest_acc the way ttnn.typecast does."""

    in_fmt = formats.input_format
    out_fmt = formats.output_format
    fp32_dest_acc_en = (
        _preserve_fp32_precision(formats)
        or out_fmt in (DataFormat.UInt32, DataFormat.Int32, DataFormat.Float32)
        or in_fmt in (DataFormat.UInt32, DataFormat.Int32)
        or out_fmt == DataFormat.Fp8_e4m3
        or in_fmt == DataFormat.Fp8_e4m3
    )
    return [DestAccumulation.Yes if fp32_dest_acc_en else DestAccumulation.No]


@parametrize(
    formats=TYPECAST_PAIRS,
    dest_acc=_production_dest_acc,
    approx_mode=[ApproximationMode.No],
    input_dimensions=[
        [32, 32]
    ],  # no need for larger tiles, as the SFPU typecast is elementwise
)
def test_eltwise_unary_typecast(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
):
    # Stimuli selection per input/output dtype:
    #  * integer -> block-float: small ints (0..15) so int->fp16b->bfp is exact
    #    (full-range ints would differ from the golden by >1 bfp ULP);
    #  * integer -> anything else: bounded range (0..255) to avoid 0xFFFF values
    #    that pack into 0xFFFFFFFF on readback, which BH NOC treats as a timeout;
    #  * float / block-float input: whole numbers so int conversions are exact and
    #    bf16 is lossless; small range when a block-float is involved so the
    #    shared-exponent quantization is (near-)exact per 16-elem block.
    bfp_involved = (
        formats.input_format.is_block_float() or formats.output_format.is_block_float()
    )
    if formats.input_format.is_integer():
        spec_A = (
            StimuliSpec.uniform(0, 15) if bfp_involved else StimuliSpec.uniform(0, 255)
        )
    else:
        spec_A = _whole_number_float_spec(16 if bfp_involved else 201)

    _run_typecast(formats, dest_acc, approx_mode, input_dimensions, spec_A)


@parametrize(
    formats=[
        pair
        for pair in TYPECAST_PAIRS
        if pair.input_format == DataFormat.UInt32
        and pair.output_format == DataFormat.Float32
    ],
    dest_acc=_production_dest_acc,
    approx_mode=[ApproximationMode.No],
    input_dimensions=[[32, 32], [32, 256]],
)
def test_eltwise_unary_typecast_uint32_to_fp32_rounding(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
):
    # Cover every uint32 binade requiring rounding, both midpoint parities,
    # and the 23-bit split boundaries. 2164260993 reproduces double rounding.
    boundaries = [0, 1, 2**32 - 1, 2164260993]
    for exponent in range(24, 32):
        ulp = 1 << (exponent - 23)
        for offset in (0, ulp, 1 << 23):
            midpoint = (1 << exponent) + offset + ulp // 2
            boundaries.extend((midpoint - 1, midpoint, midpoint + 1))
    for high in (1, 2, 127, 128, 255, 256, 510, 511):
        boundaries.extend(((high << 23) - 1, high << 23, (high << 23) + 1))

    def distribution(size, dtype, generator):
        # Keep integers exact until conversion by the device / golden generator.
        values = torch.randint(
            0, 2**32, (size,), dtype=torch.int64, generator=generator
        )
        values[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int64)
        return values.to(dtype)

    _run_typecast(
        formats,
        dest_acc,
        approx_mode,
        input_dimensions,
        StimuliSpec(distribution=distribution, seed=52787),
        max_ulp=0,
    )


_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1


def _uint_to_uint16_saturation_boundaries(input_format: DataFormat) -> list[int]:
    """Inputs on both sides of the UInt16 saturation edge, plus bit-31-set values."""
    if input_format == DataFormat.UInt8:
        return list(range(256))
    boundaries = [
        0,
        1,
        _UINT16_MAX - 1,
        _UINT16_MAX,
        _UINT16_MAX + 1,
        _UINT16_MAX + 2,
        1 << 17,
        (1 << 31) - 1,
        1 << 31,
        (1 << 31) + 1,
        _UINT32_MAX - 1,
        _UINT32_MAX,
    ]
    for bit in range(15, 32):
        boundaries.extend(((1 << bit) - 1, 1 << bit, (1 << bit) + 1))
    return boundaries


@parametrize(
    formats=[
        pair
        for pair in TYPECAST_PAIRS
        if pair.input_format in (DataFormat.UInt32, DataFormat.UInt8)
        and pair.output_format == DataFormat.UInt16
    ],
    dest_acc=_production_dest_acc,
)
def test_eltwise_unary_typecast_uint_to_uint16_saturation(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
):
    # The functional matrix feeds uniform(0, 255), so it never reaches the
    # saturating lanes; pin x >= 2^16 and bit-31-set inputs to min(x, 65535).
    input_dimensions = [32, 32]
    boundaries = _uint_to_uint16_saturation_boundaries(formats.input_format)

    def distribution(size, dtype, generator):
        # Called once per face. Boundaries take the even lanes; every odd lane
        # stays below 65535 so no packed 32-bit word of the UInt16 result reads
        # 0xFFFFFFFF (which BH NOC treats as a readback timeout). UInt8 inputs
        # cannot saturate, so all 256 of them fill the face directly.
        if formats.input_format == DataFormat.UInt8:
            return torch.tensor(boundaries, dtype=torch.int64)[:size].to(dtype)
        values = torch.randint(
            0, 1 << 32, (size,), dtype=torch.int64, generator=generator
        )
        values[1::2] = torch.randint(
            0, _UINT16_MAX, (size // 2,), dtype=torch.int64, generator=generator
        )
        assert 2 * len(boundaries) <= size
        values[: 2 * len(boundaries) : 2] = torch.tensor(boundaries, dtype=torch.int64)
        return values.to(dtype)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec(distribution=distribution, seed=62824),
        spec_B=StimuliSpec(distribution=distribution, seed=62824),
    )
    src_values = [int(v) & _UINT32_MAX for v in src_A.flatten().tolist()]
    golden = [min(v, _UINT16_MAX) for v in src_values]

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        ApproximationMode.No,
        input_dimensions,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
    )
    result = [int(v) & _UINT16_MAX for v in configuration.run().result]
    assert len(result) == len(
        golden
    ), f"expected {len(golden)} result elements, got {len(result)}"
    assert set(boundaries) <= set(src_values), "boundary inputs missing from stimuli"

    mismatches = [(x, g, r) for x, g, r in zip(src_values, golden, result) if g != r]
    assert not mismatches, (
        f"{len(mismatches)} {formats.input_format.name}->UInt16 mismatches; first "
        f"(input, expected, got): {[(hex(x), g, r) for x, g, r in mismatches[:16]]}"
    )


# The uint8 wrap and uint16 saturation edges the 0..255 sweep above never reaches. Odd lanes stay in
# 0..127 so no readback word is all-ones, which is what the Blackhole NOC returns on a timeout.
_INT_EDGE_PAIRS = [
    InputOutputFormat(DataFormat.Int32, DataFormat.UInt8),
    InputOutputFormat(DataFormat.UInt32, DataFormat.UInt8),
    InputOutputFormat(DataFormat.UInt16, DataFormat.UInt8),
    InputOutputFormat(DataFormat.UInt32, DataFormat.UInt16),
]
_INT_EDGE_VALUES = {
    DataFormat.Int32: (
        [0, 1, -1, -2, 127, -127, 128, -128, -129, 255, -255, 256, -256, -257, 1000, -1000,
         65535, -65535, 65536, -65536, 2**31 - 1, -(2**31) + 1, -(2**31)],
        -(2**31),
        2**31,
    ),
    DataFormat.UInt32: (
        [0, 1, 127, 128, 255, 256, 257, 65534, 65535, 65536, 65537, 2**24, 2**31 - 1, 2**31,
         2**31 + 65535, 2**32 - 2, 2**32 - 1],
        0,
        2**32,
    ),
    DataFormat.UInt16: (
        [0, 1, 127, 128, 255, 256, 257, 32767, 32768, 65534, 65535],
        0,
        2**16,
    ),
}  # fmt: skip


def _int_edge_spec(in_fmt: DataFormat) -> StimuliSpec:
    edges, low, high = _INT_EDGE_VALUES[in_fmt]

    def distribution(size, dtype, generator):
        values = torch.randint(
            low, high, (size,), dtype=torch.int64, generator=generator
        )
        values[0 : 2 * len(edges) : 2] = torch.tensor(edges, dtype=torch.int64)
        values[1::2] &= 0x7F
        return values.to(dtype)

    return StimuliSpec(distribution=distribution, seed=0)


@parametrize(
    formats=_INT_EDGE_PAIRS,
    dest_acc=_production_dest_acc,
)
def test_eltwise_unary_typecast_int_edges(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
):
    _run_typecast(
        formats,
        dest_acc,
        ApproximationMode.No,
        [32, 32],
        _int_edge_spec(formats.input_format),
        twos_complement=formats.input_format == DataFormat.Int32,
    )


_FP32_EXP_BIAS = 127
_FP32_MANTISSA_BITS = 23
_FP32_SIGN_BIT = 1 << 31
_UINT32_WRAP = 1 << 32


def _fp32_bits(unbiased_exp: int, mantissa: int) -> int:
    """Positive fp32 bit pattern for 1.mantissa * 2^unbiased_exp."""
    return ((unbiased_exp + _FP32_EXP_BIAS) << _FP32_MANTISSA_BITS) | mantissa


def _fp32_unbiased_exp(bits: int) -> int:
    return ((bits >> _FP32_MANTISSA_BITS) & 0xFF) - _FP32_EXP_BIAS


def _fp32_to_uint8_wrap(bits: int) -> int:
    """Exact uint8 wrap of an fp32 bit pattern: trunc toward zero, low byte.

    Python ints keep every binade exact (the shared golden goes through int64,
    which overflows from 2^63 up); inf/NaN produce 0, as the kernel does.
    """
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        return 0
    return math.trunc(value) % 256


def _fp32_to_uint8_wide_exponent_bits() -> list[int]:
    """fp32 bit patterns covering every binade at or above 2^24 with nonzero low
    mantissa bits, both signs, plus the issue's two named inputs and small values."""
    patterns = [
        _fp32_bits(
            55, 0x000001
        ),  # 2^55 + 2^32 (shift amount 32 wraps to 0 -> low byte 1)
        _fp32_bits(60, 0x000001),  # 2^60 + 2^37
    ]
    mantissas = (0x000001, 0x0000FF, 0x00007F, 0x123457, 0x5A5A5B, 0x7FFFFF)
    for unbiased_exp in range(24, 128):
        for mantissa in mantissas:
            patterns.append(_fp32_bits(unbiased_exp, mantissa))
    patterns += [0x7F800000, 0x7FC00000]  # +inf, NaN
    # Small whole and fractional values keep the in-range path pinned too.
    for unbiased_exp in range(-2, 24):
        for mantissa in (0x000000, 0x400001, 0x7FFFFF):
            patterns.append(_fp32_bits(unbiased_exp, mantissa))
    patterns += [p | _FP32_SIGN_BIT for p in patterns]
    return patterns


def test_eltwise_unary_typecast_fp32_to_uint8_wide_exponent():
    # SFPSHFT uses (amount & 31); without a bound on the large side, exponents
    # 55..62, 87..94 and 119..126 shift the mantissa back into the low byte.
    # Every finite |x| >= 2^31 is a multiple of 256, so the wrapped result must be 0.
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.UInt8)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [32, 64]
    patterns = _fp32_to_uint8_wide_exponent_bits()

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    # The stimuli generator calls a custom distribution once per face, so place
    # the bit patterns over the whole operand afterwards.
    assert len(patterns) <= src_A.numel()
    bits = patterns + [0] * (src_A.numel() - len(patterns))
    src_A = (
        torch.tensor(
            [b - _UINT32_WRAP if b & _FP32_SIGN_BIT else b for b in bits],
            dtype=torch.int32,
        )
        .view(torch.float32)
        .to(src_A.dtype)
    )
    src_bits = [
        b & 0xFFFFFFFF for b in src_A.to(torch.float32).view(torch.int32).tolist()
    ]
    golden = [_fp32_to_uint8_wrap(b) for b in src_bits]

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        ApproximationMode.No,
        input_dimensions,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
    )
    result = [int(v) & 0xFF for v in configuration.run().result]
    assert len(result) == len(
        golden
    ), f"expected {len(golden)} result elements, got {len(result)}"

    mismatches = [(b, g, r) for b, g, r in zip(src_bits, golden, result) if g != r]
    failing_exponents = sorted({_fp32_unbiased_exp(b) for b, _, _ in mismatches})
    assert not mismatches, (
        f"{len(mismatches)} fp32->uint8 mismatches at unbiased exponents "
        f"{failing_exponents}; first (input bits, expected, got): "
        f"{[(hex(b), g, r) for b, g, r in mismatches[:16]]}"
    )


# 32 tiles per run (the largest operand other LLK tests use); full-range UInt16 takes two.
_UINT_TO_UINT8_DIMENSIONS = [128, 256]
_UINT_TO_UINT8_CHUNK = _UINT_TO_UINT8_DIMENSIONS[0] * _UINT_TO_UINT8_DIMENSIONS[1]


def _uint_to_uint8_wide_range_values(input_format: DataFormat, chunk: int) -> list[int]:
    """Inputs that set the bits above bit 7, which the kernel must drop.

    UInt16 is swept exhaustively (``chunk`` selects the half). Int32/UInt32 get
    every 2^k - 1, 2^k, 2^k + 1, the 256-wrap points, the top of the range and a
    random fill. Int32 stays non-negative: Wormhole Dest holds Int32 as
    sign-magnitude, so a negative input reads back as |x| & 0xFF there.
    """
    if input_format == DataFormat.UInt16:
        start = chunk * _UINT_TO_UINT8_CHUNK
        return list(range(start, start + _UINT_TO_UINT8_CHUNK))

    top = 2**32 if input_format == DataFormat.UInt32 else 2**31
    values = [0, 1, 254, 255, 256, 257, 511, 512, 0xFF00, 0xFFFF, 0x10000, 0x100FF]
    for k in range(1, 32):
        values += [v for v in ((1 << k) - 1, 1 << k, (1 << k) + 1) if v < top]
    values += list(range(top - 256, top))
    generator = torch.Generator().manual_seed(62765)
    fill = _UINT_TO_UINT8_CHUNK - len(values)
    values += torch.randint(
        0, top, (fill,), dtype=torch.int64, generator=generator
    ).tolist()
    return values


@parametrize(
    formats=[
        InputOutputFormat(in_fmt, DataFormat.UInt8)
        for in_fmt in (DataFormat.Int32, DataFormat.UInt32, DataFormat.UInt16)
        if in_fmt in _ARCH_SUPPORTED_FORMATS
    ],
    chunk=lambda formats: [0, 1] if formats.input_format == DataFormat.UInt16 else [0],
)
def test_eltwise_unary_typecast_uint_to_uint8_wide_range(
    formats: InputOutputFormat, chunk: int
):
    # The kernel keeps only the low byte. The default stimuli (0..255) cannot tell
    # x & 0xFF from x, (x + 256) & 0xFF or (x & 0xFFFF) & 0xFF, so drive the
    # bits above bit 7 here.
    dest_acc = _production_dest_acc(formats)[0]
    input_dimensions = _UINT_TO_UINT8_DIMENSIONS
    values = _uint_to_uint8_wide_range_values(formats.input_format, chunk)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    # A custom distribution is called once per face, so overwrite the operand.
    assert len(values) == src_A.numel()
    src_A = torch.tensor(values, dtype=torch.int64).to(src_A.dtype)
    golden = [v & 0xFF for v in values]

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        ApproximationMode.No,
        input_dimensions,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
    )
    result = [int(v) & 0xFF for v in configuration.run().result]
    assert len(result) == len(
        golden
    ), f"expected {len(golden)} result elements, got {len(result)}"

    mismatches = [(v, g, r) for v, g, r in zip(values, golden, result) if g != r]
    assert not mismatches, (
        f"{len(mismatches)} {formats.input_format.name}->UInt8 mismatches; first "
        f"(input, expected, got): {[(hex(v), g, r) for v, g, r in mismatches[:16]]}"
    )


def _build_typecast_config(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
    src_A: torch.Tensor,
    tile_cnt_A: int,
    src_B: torch.Tensor,
    tile_cnt_B: int,
    twos_complement: bool = False,
) -> TestConfig:
    # Unpack straight into Dest when either:
    #  * the input is 32-bit -- the unpacker has no SrcA/SrcB path for
    #    Int32/UInt32/Float32, so is_unpacker_format_conversion_supported_dest
    #    (cunpack_common.h) asserts unpack_to_dest for them; or
    #  * production would, i.e. preserve_fp32_precision drives UnpackToDestFp32 in
    #    the ttnn program factories.
    # For non-32-bit inputs the unpack MOP still gates on is_32bit_input(), so the
    # flag only actually changes the datapath for genuine 32-bit inputs.
    unpack_to_dest = formats.input_format.is_32_bit() or _preserve_fp32_precision(
        formats
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_typecast_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
            # Emits SFPU_UNARY_OPERATION = SfpuType::typecast so the kernel goes
            # through the shared unary-SFPU dispatch; TYPECAST_FORMATS supplies the
            # (input, output) pair that selects the concrete typecast kernel.
            MATH_OP(mathop=MathOperation.Typecast),
            TYPECAST_FORMATS(formats.input_format, formats.output_format),
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
        unpack_to_dest=unpack_to_dest,
    )

    # pack_src fix for the SFPU typecast.
    #
    # The generic format inference models pack_src from the *input* (what the
    # unpacker writes to Dest). For a typecast the SFPU overwrites Dest with the
    # *output* value, so the packer must read the output's register
    # representation, not the input's. In 16-bit Dest mode (dest_acc=No) the
    # inferred pack_src would stay equal to the input format and the pack would
    # be rejected (e.g. UInt16 -> Float16_b is not a supported packer
    # conversion). Patch pack_src to the format the SFPU actually leaves in Dest:
    # block-float outputs are produced as Float16_b and compressed by the packer;
    # every other output is already in its packable register form. (For
    # dest_acc=Yes the 32-bit gasket converts from Dest, and the inference
    # already yields the output format, so no patch is needed there.)
    if dest_acc == DestAccumulation.No:
        pack_src = (
            DataFormat.Float16_b
            if formats.output_format.is_block_float()
            else formats.output_format
        )
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = pack_src

    return configuration


def _run_typecast(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
    spec_A: StimuliSpec,
    *,
    max_ulp: int | None = None,
    twos_complement: bool = False,
):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
        spec_B=spec_A,
    )

    generate_golden = get_golden_generator(TypecastGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.input_format,
        formats.output_format,
        input_dimensions,
    )

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        approx_mode,
        input_dimensions,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
        twos_complement,
    )

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, max_ulp=max_ulp
    ), "Assert against golden failed"
