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
    format_dict,
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
from helpers.utils import passed_test

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

_BLOCK_FLOAT_FORMATS = (DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b)

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


def _is_block_float(fmt: DataFormat) -> bool:
    return fmt in _BLOCK_FLOAT_FORMATS


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


def _matrix_spec(formats: InputOutputFormat) -> StimuliSpec:
    """Stimuli for one (input, output) pair of the typecast matrix.

    * integer -> block-float: small ints (0..15) so int->fp16b->bfp is exact
      (full-range ints would differ from the golden by >1 bfp ULP);
    * integer -> anything else: bounded range (0..255) to avoid 0xFFFF values
      that pack into 0xFFFFFFFF on readback, which BH NOC treats as a timeout;
    * float / block-float input: whole numbers so int conversions are exact and
      bf16 is lossless; small range when a block-float is involved so the
      shared-exponent quantization is (near-)exact per 16-elem block.
    """
    bfp_involved = _is_block_float(formats.input_format) or _is_block_float(
        formats.output_format
    )
    if formats.input_format.is_integer():
        return (
            StimuliSpec.uniform(0, 15) if bfp_involved else StimuliSpec.uniform(0, 255)
        )
    return _whole_number_float_spec(16 if bfp_involved else 201)


def _uint16_full_range_spec() -> StimuliSpec:
    """Every UInt16 value, with the bf16-inexact boundary and the extremes pinned.

    256 is the last integer bfloat16 holds exactly; 257, 0x8000 and 0xFFFF have to
    be rounded, so the int -> bf16 rounding is exercised and not just the load. (The
    matrix caps integer inputs at 255 for the sake of its integer outputs.)
    """
    pinned = torch.tensor([256, 257, 0x8000, 0xFFFF])

    def dist(size, dtype, generator):
        values = torch.randint(0, 1 << 16, (size,), generator=generator)
        values[: len(pinned)] = pinned
        return values.to(dtype)

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
    _run_typecast(
        formats, dest_acc, approx_mode, input_dimensions, _matrix_spec(formats)
    )


@parametrize(
    formats=[
        pair
        for pair in TYPECAST_PAIRS
        if pair.input_format == DataFormat.UInt16
        and pair.output_format
        in (DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b)
    ],
    dest_acc=[DestAccumulation.Yes],
    approx_mode=[ApproximationMode.No],
    input_dimensions=[[32, 32]],
)
def test_eltwise_unary_typecast_uint16_to_fp16b_32bit_dest(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
):
    # ttnn.typecast runs this pair in 16-bit Dest (the matrix above pins dest_acc
    # to No), but the compute API allows it with fp32_dest_acc_en too. In 32-bit
    # Dest the UInt16 datum occupies the low half of the 32-bit word, so a 16-bit
    # Dest load reads the other half; the kernel has to load the whole word and
    # mask it, as every other UInt16-input typecast does.
    #
    # Float16_b output takes the full UInt16 range so the bf16 rounding of values
    # above 256 is exercised too; the block-float outputs keep the matrix's small
    # ints (large ints would differ from the golden by more than 1 bfp ULP).
    spec_A = (
        _uint16_full_range_spec()
        if formats.output_format == DataFormat.Float16_b
        else _matrix_spec(formats)
    )
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


_FP32_EXP_MASK = 0xFF << _FP32_MANTISSA_BITS


def _fp32_is_nan(bits: int) -> bool:
    return (bits & _FP32_EXP_MASK) == _FP32_EXP_MASK and bits & (
        (1 << _FP32_MANTISSA_BITS) - 1
    )


def _fp32_to_fp16b_special_bits() -> list[int]:
    """fp32 bit patterns around the Float32 -> Float16_b rounding step.

    The kernel rounds to nearest-even on the bit pattern (+0x7FFF + lsb, then clear
    the low 16 bits), which is only valid for finite values: on a NaN the add can
    carry out of the mantissa (0x7FFFFFFF -> 0x80000000, a signed zero) and the mask
    can clear every mantissa bit a low-payload NaN had (0x7F800001 -> +inf).
    """
    patterns = [
        0x7F800001,  # NaN, payload below bit 16: the mask alone leaves +inf
        0x7F808000,  # NaN, payload bit 15: +0x7FFF carries it out, the mask leaves +inf
        0x7FFFFFFF,  # NaN, every mantissa bit: the carry runs through the sign (-0.0)
        0x7FFF8000,  # NaN, bits 15..22: same carry (lsb set) -> -0.0
        0x7FC00000,  # canonical quiet NaN: unaffected today and must stay NaN
        0x7F800000,  # +inf must stay +inf (exponent 255, zero mantissa)
        0x7F7FFFFF,  # largest finite rounds up to +inf, as float32 -> bfloat16 does
        0x3F808000,  # 1 + 2^-8, a tie: rounds to even (1.0)
        0x3F818000,  # 1 + 2^-7 + 2^-8, a tie: rounds to even (1 + 2^-6)
        0x3F80FFFF,  # just above the tie: rounds up
        0x3F807FFF,  # just below the tie: rounds down
        0x00800000,  # smallest normal
        0x00000000,  # +0
    ]
    return patterns + [p | _FP32_SIGN_BIT for p in patterns]


def _fp32_random_normal_bits(count: int, seed: int) -> list[int]:
    """Random normal fp32 bit patterns (both signs, exponents 1..254, any mantissa).

    Exponent 0 is left out because Dest flushes denormals; exponent 255 is covered
    by the special patterns.
    """
    generator = torch.Generator().manual_seed(seed)
    sign = torch.randint(0, 2, (count,), generator=generator, dtype=torch.int64) << 31
    exponent = (
        torch.randint(1, 255, (count,), generator=generator, dtype=torch.int64)
        << _FP32_MANTISSA_BITS
    )
    mantissa = torch.randint(
        0, 1 << _FP32_MANTISSA_BITS, (count,), generator=generator, dtype=torch.int64
    )
    return (sign | exponent | mantissa).tolist()


def test_eltwise_unary_typecast_fp32_to_fp16b_rounding_specials():
    # The SFPU half of Float32 -> Float16_b rounds the fp32 word in Dest to bfloat16
    # precision and stores it back as fp32; the packer narrows it afterwards. Read
    # that fp32 word back as it is (Float32 L1 output): the packer's own fp32 -> bf16
    # conversion maps NaN to infinity, so a bf16 output could not show what the SFPU
    # left in Dest.
    kernel_formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b)
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes  # a Float32 input needs 32-bit Dest
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    specials = _fp32_to_fp16b_special_bits()
    assert len(specials) <= src_A.numel()
    bits = specials + _fp32_random_normal_bits(src_A.numel() - len(specials), seed=7301)
    src_A = torch.tensor(
        [b - _UINT32_WRAP if b & _FP32_SIGN_BIT else b for b in bits],
        dtype=torch.int32,
    ).view(torch.float32)
    # torch rounds float32 -> bfloat16 to nearest-even and overflows to inf, like the
    # kernel; it canonicalises NaN (dropping the sign), so NaN lanes are checked apart.
    golden_bits = [
        g & 0xFFFFFFFF
        for g in src_A.to(torch.bfloat16).to(torch.float32).view(torch.int32).tolist()
    ]

    configuration = _build_typecast_config(
        formats,
        dest_acc,
        ApproximationMode.No,
        input_dimensions,
        src_A,
        tile_cnt_A,
        src_B,
        tile_cnt_B,
        kernel_formats=kernel_formats,
    )
    result = configuration.run().result
    assert len(result) == len(
        bits
    ), f"expected {len(bits)} result elements, got {len(result)}"

    mismatches = []
    for b, g, r in zip(bits, golden_bits, result):
        got = struct.unpack("<I", struct.pack("<f", r))[0]
        if _fp32_is_nan(b):
            # NaN in, NaN out with the input sign and the low 16 bits cleared like
            # every other lane; the payload above them is not pinned.
            ok = (
                math.isnan(r)
                and (math.copysign(1.0, r) < 0) == bool(b & _FP32_SIGN_BIT)
                and (got & 0xFFFF) == 0
            )
            expected = "nan" if not b & _FP32_SIGN_BIT else "-nan"
        else:
            ok = got == g
            expected = hex(g)
        if not ok:
            mismatches.append((hex(b), expected, hex(got)))
    assert not mismatches, (
        f"{len(mismatches)} fp32->bf16 mismatches (input bits, expected, got): "
        f"{mismatches[:16]}"
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
    kernel_formats: InputOutputFormat | None = None,
) -> TestConfig:
    """Build the test config for an L1 ``formats`` pair.

    ``kernel_formats`` selects the SFPU typecast kernel (``typecast_tile<IN, OUT>``)
    when it must differ from the L1 pair; by default the kernel pair is ``formats``.
    """
    kernel_formats = kernel_formats or formats
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
            TYPECAST_FORMATS(kernel_formats.input_format, kernel_formats.output_format),
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
            if _is_block_float(formats.output_format)
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
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, max_ulp=max_ulp
    ), "Assert against golden failed"
