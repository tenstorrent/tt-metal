# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK SFPU typecast tests.

Scope: the **full ttnn typecast matrix** — every directed dtype pair that
``ttnn.typecast`` exercises end-to-end. That spans float<->float, float<->int,
int<->int and all block-float (Bfp8_b / Bfp4_b) conversions. Same-dtype pairs
and the ``int32<->uint32`` pair (not a kernel pair) are excluded.
"""

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
    bfp_involved = _is_block_float(formats.input_format) or _is_block_float(
        formats.output_format
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


# The pairs whose 32-bit Dest bodies run SFPLOADMACRO programs on Blackhole, with the 16-bit Dest
# pair that shares the fp32 -> uint16 macro.
_MACRO_PAIRS_32B = [
    (DataFormat.UInt16, DataFormat.Float32),
    (DataFormat.UInt16, DataFormat.UInt32),
    (DataFormat.UInt16, DataFormat.Int32),
    (DataFormat.Float32, DataFormat.UInt16),
    (DataFormat.Float16_b, DataFormat.UInt16),
    (DataFormat.Float32, DataFormat.Float16_b),
]


def _edge_spec_uint16() -> StimuliSpec:
    # Every uint16 value class, then random values. 0xFFFF stays on even positions so no 32-bit
    # output word reads back as all ones (which the readback path treats as a timeout).
    def dist(size, dtype, generator):
        values = torch.randint(0, 65535, (size,), generator=generator)
        fixed = torch.tensor(
            [0, 1, 255, 256, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF, 0, 0xFFFF, 0, 2, 0xFFFF, 0]
        )
        values[: len(fixed)] = fixed
        values[1::2] = torch.minimum(values[1::2], torch.tensor(0xFFFE))
        return values.to(dtype)

    return StimuliSpec(distribution=dist, seed=16)


def _edge_spec_float_to_uint16() -> StimuliSpec:
    # Negatives, fractions with ties, the top of the range and past it, then random values; the
    # saturating values stay off the odd positions.
    def dist(size, dtype, generator):
        values = torch.rand(size, generator=generator) * 70100.0 - 100.0
        fixed = torch.tensor(
            [
                0.0,
                -0.0,
                -1.0,
                -0.5,
                0.5,
                1.5,
                2.5,
                3.5,
                254.5,
                255.5,
                65534.0,
                65534.5,
                65535.0,
                0.0,
                70000.0,
                0.0,
                1.0e9,
                0.0,
                -1.0e9,
                0.0,
            ]
        )
        values[: len(fixed)] = fixed
        values[1::2] = torch.clamp(values[1::2], max=65534.0)
        return values.to(dtype)

    return StimuliSpec(distribution=dist, seed=32)


def _edge_spec_float_to_bf16() -> StimuliSpec:
    # Low 16 bits on and around the rounding tie, plus lanes with a random low half.
    def dist(size, dtype, generator):
        exponent = torch.randint(
            100, 150, (size,), generator=generator, dtype=torch.int32
        )
        mantissa = torch.randint(
            0, 1 << 23, (size,), generator=generator, dtype=torch.int32
        )
        sign = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int32)
        low = torch.tensor(
            [0x8000, 0x7FFF, 0x8001, 0x0000, 0xFFFF, 0x8000, 0x8000, 0x7FFF]
            * (size // 8 + 1)
        )[:size]
        mantissa = (mantissa & 0x7F0000) | low.to(torch.int32)
        mask = (torch.arange(size) % 4) == 3
        mantissa[mask] = torch.randint(
            0, 1 << 23, (int(mask.sum()),), generator=generator, dtype=torch.int32
        )
        bits = (sign << 31) | (exponent << 23) | mantissa
        return bits.view(torch.float32).to(dtype)

    return StimuliSpec(distribution=dist, seed=64)


def _as_float_tensor(src) -> torch.Tensor:
    return (
        src.float()
        if isinstance(src, torch.Tensor)
        else torch.tensor(src, dtype=torch.float32)
    )


def _golden_float_to_uint16(src):
    # Clamp at zero, round to nearest with ties away from zero (SFPSTOCHRND float to integer),
    # saturate at 65535; floor(x + 0.5) in float64 is exact for every float32 x.
    values = torch.clamp(_as_float_tensor(src).to(torch.float64), min=0.0)
    return torch.clamp(torch.floor(values + 0.5), 0, 65535).to(torch.int32).flatten()


@parametrize(
    formats=[
        InputOutputFormat(i, o)
        for i, o in _MACRO_PAIRS_32B
        if InputOutputFormat(i, o) in TYPECAST_PAIRS
    ],
    dest_acc=_production_dest_acc,
    approx_mode=[ApproximationMode.No],
    input_dimensions=[[32, 64]],
)
def test_eltwise_unary_typecast_macro_pairs_edges(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
):
    in_fmt, out_fmt = formats.input_format, formats.output_format
    golden_fn = None
    if in_fmt == DataFormat.UInt16:
        spec = _edge_spec_uint16()
    elif out_fmt == DataFormat.UInt16:
        spec = _edge_spec_float_to_uint16()
        golden_fn = _golden_float_to_uint16
    else:
        spec = _edge_spec_float_to_bf16()
    _run_typecast(
        formats,
        dest_acc,
        approx_mode,
        input_dimensions,
        spec,
        max_ulp=0,
        twos_complement=(out_fmt == DataFormat.Int32),
        golden_fn=golden_fn,
    )


_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _golden_float_to_int32(src):
    # Truncation toward zero, saturation by the sign (NaN by its sign bit), zero and denormals to 0.
    values = _as_float_tensor(src)
    negative = values.view(torch.int32) < 0
    magnitude = values.abs()
    out = torch.trunc(values).to(torch.float64)
    out = torch.where(magnitude < 1.0, torch.zeros_like(out), out)
    saturate = torch.isnan(values) | (magnitude >= 2.0**31)
    out = torch.where(
        saturate & ~negative, torch.full_like(out, float(_INT32_MAX)), out
    )
    out = torch.where(saturate & negative, torch.full_like(out, float(_INT32_MIN)), out)
    return out.to(torch.int64).to(torch.int32).flatten()


def _edge_spec_float_to_int32(input_format: DataFormat) -> StimuliSpec:
    # Both signs of zero, denormals, ties, the 2^24 and 2^31 boundaries, inf and NaN, then random.
    def dist(size, dtype, generator):
        fixed_bits = [
            0x00000000,
            0x80000000,  # +0, -0
            0x00000001,
            0x80000001,
            0x007FFFFF,  # denormals
            0x3F7FFFFF,
            0xBF7FFFFF,  # 1 - ulp
            0x3F800000,
            0xBF800000,  # 1
            0x3FC00000,
            0xBFC00000,  # 1.5
            0x40200000,
            0xC0200000,  # 2.5
            0x4B7FFFFF,
            0xCB7FFFFF,  # 2^24 - 1
            0x4B800000,
            0xCB800000,  # 2^24
            0x4EFFFFFF,
            0xCEFFFFFF,  # 2^31 - 128 (the largest magnitude below 2^31)
            0x4F000000,
            0xCF000000,  # 2^31
            0x4F000001,
            0xCF000001,  # 2^31 + 256
            0x501502F9,
            0xD01502F9,  # 1e10
            0x7F7FFFFF,
            0xFF7FFFFF,  # max finite
            0x7F800000,
            0xFF800000,  # inf
            0x7FC00000,
            0xFFC00000,
            0x7F800001,
            0xFF800001,
            0x7FFFFFFF,
            0xFFFFFFFF,  # NaN payloads
            0x3E800000,
            0xBE800000,  # 0.25
            0x42F6E979,
            0xC2F6E979,  # 123.456
        ]
        exponent = torch.randint(
            100, 160, (size,), generator=generator, dtype=torch.int64
        )
        mantissa = torch.randint(
            0, 1 << 23, (size,), generator=generator, dtype=torch.int64
        )
        sign = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
        bits = (sign << 31) | (exponent << 23) | mantissa
        bits[: len(fixed_bits)] = torch.tensor(fixed_bits, dtype=torch.int64)
        if input_format != DataFormat.Float32:
            # a bfloat16 input carries the high 16 bits only
            bits = bits & 0xFFFF0000
        return bits.to(torch.int32).view(torch.float32).to(dtype)

    return StimuliSpec(distribution=dist, seed=2**31 - 1)


@parametrize(
    formats=[
        pair
        for pair in TYPECAST_PAIRS
        if pair.output_format == DataFormat.Int32
        and pair.input_format in (DataFormat.Float32, DataFormat.Float16_b)
    ],
    dest_acc=_production_dest_acc,
    approx_mode=[ApproximationMode.No],
    input_dimensions=[[32, 64]],
)
def test_eltwise_unary_typecast_float_to_int32_edges(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
):
    _run_typecast(
        formats,
        dest_acc,
        approx_mode,
        input_dimensions,
        _edge_spec_float_to_int32(formats.input_format),
        max_ulp=0,
        twos_complement=True,
        golden_fn=_golden_float_to_int32,
    )


def _run_typecast(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
    spec_A: StimuliSpec,
    *,
    max_ulp: int | None = None,
    twos_complement: bool = False,
    golden_fn=None,
):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
        spec_B=spec_A,
    )

    if golden_fn is None:
        generate_golden = get_golden_generator(TypecastGolden)
        golden_tensor = generate_golden(
            src_A,
            formats.input_format,
            formats.output_format,
            input_dimensions,
        )
    else:
        golden_tensor = golden_fn(src_A)

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
            if _is_block_float(formats.output_format)
            else formats.output_format
        )
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = pack_src

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    if max_ulp == 0 and formats.output_format.is_integer():
        # The harness ULP gate covers the float formats only; integer outputs compare value for value.
        golden_i = golden_tensor.flatten().to(torch.int64)
        result_i = res_tensor.flatten().to(torch.int64)
        mismatch = (golden_i != result_i).nonzero().flatten()
        assert mismatch.numel() == 0, (
            f"{mismatch.numel()} of {golden_i.numel()} results differ from the golden; first at index "
            f"{int(mismatch[0])}: input {src_A.flatten()[int(mismatch[0])].item()}, golden "
            f"{int(golden_i[mismatch[0]])}, result {int(result_i[mismatch[0]])}"
        )
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format, max_ulp=max_ulp
        ), "Assert against golden failed"
