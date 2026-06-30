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

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
