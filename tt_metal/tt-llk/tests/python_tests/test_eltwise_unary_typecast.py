# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK SFPU typecast tests.

Scope: the **full ttnn typecast matrix** — every directed dtype pair that
``ttnn.typecast`` exercises end-to-end. That spans float<->float, float<->int,
int<->int and all block-float (Bfp8_b / Bfp4_b) conversions. Same-dtype pairs
and the ``int32<->uint32`` pair (not a kernel pair) are excluded.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    TypecastGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
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

# Every directed pair (54 total) ttnn.typecast exercises end-to-end.
TYPECAST_PAIRS = [
    InputOutputFormat(in_fmt, out_fmt)
    for in_fmt in _TYPECAST_FORMATS
    for out_fmt in _TYPECAST_FORMATS
    if in_fmt != out_fmt and (in_fmt, out_fmt) not in _EXCLUDED_PAIRS
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


def _required_dest_acc(formats: InputOutputFormat) -> DestAccumulation:
    """32-bit Dest is mandatory whenever the SFPU integer datapath is used.

    The SFPU typecast computes the result into Dest before the packer reads it,
    and the integer datapath always operates on 32-bit Dest data. So any integer
    input or output (not just the 32-bit ones) requires ``dest_acc=Yes`` —
    otherwise pack_src stays 16-bit and ``is_packer_to_L1_conversion_supported``
    rejects the pack (LLK assert in configure_pack). 32-bit float formats also
    require 32-bit Dest.
    """
    in_fmt = formats.input_format
    out_fmt = formats.output_format
    if (
        in_fmt.is_integer()
        or out_fmt.is_integer()
        or in_fmt.is_32_bit()
        or out_fmt.is_32_bit()
    ):
        return DestAccumulation.Yes
    return DestAccumulation.No


@parametrize(
    formats=TYPECAST_PAIRS,
    approx_mode=[ApproximationMode.No],
    input_dimensions=[
        [32, 32]
    ],  # no need for larger tiles, as the SFPU typecast is elementwise
)
def test_eltwise_unary_typecast(
    formats: InputOutputFormat,
    approx_mode: ApproximationMode,
    input_dimensions: list[int],
):
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip("Typecast SFPU test targets Blackhole and Wormhole_b0")

    arch_formats = TestConfig.DATA_FORMAT_ENUM
    if (
        formats.input_format not in arch_formats
        or formats.output_format not in arch_formats
    ):
        pytest.skip(
            f"{formats.input_format.name}->{formats.output_format.name} "
            f"not supported on {get_chip_architecture().name}"
        )

    dest_acc = _required_dest_acc(formats)

    # Stimuli selection per input/output dtype:
    #  * integer -> block-float: small ints (0..15) so int->fp16b->bfp is exact
    #    (full-range ints would differ from the golden by >1 bfp ULP);
    #  * integer -> anything else: format-aware default spec (exact int/float cmp);
    #  * float / block-float input: whole numbers so int conversions are exact and
    #    bf16 is lossless; small range when a block-float is involved so the
    #    shared-exponent quantization is (near-)exact per 16-elem block.
    bfp_involved = _is_block_float(formats.input_format) or _is_block_float(
        formats.output_format
    )
    if formats.input_format.is_integer():
        spec_A = StimuliSpec.uniform(0, 15) if bfp_involved else None
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

    # 32-bit inputs are unpacked straight into Dest (no Src-register staging).
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
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

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
