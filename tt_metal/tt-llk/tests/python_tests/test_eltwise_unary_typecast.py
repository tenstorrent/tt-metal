# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK SFPU typecast tests.

These exercise the ``calculate_typecast_*`` / ``init_typecast_*`` SFPU kernels
directly from the tt-llk harness — the coverage gap identified in
``TYPECAST_TEST_COVERAGE_ANALYSIS.md`` (the tt-llk infra previously had *zero*
typecast tests, while ttnn covered ~54 dtype pairs end-to-end).

Each test drives ``sources/eltwise_unary_typecast_test.cpp`` which mirrors the
production typecast compute kernel: copy the input tile into Dest, run the SFPU
typecast in place, then pack to the output format.

Scope: the SFPU *numeric* conversions (float<->int, int<->int, fp32->fp16b),
which is exactly what the existing datacopy tests do NOT cover. The pure
unpacker/packer block-float<->float conversions are already covered by the
datacopy tests and are intentionally excluded here.
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

# SFPU typecast pairs that perform real arithmetic in Dest (the coverage gap).
# Grouped for readability; deduplicated into TYPECAST_PAIRS below.
_FLOAT_TO_FLOAT = [
    (DataFormat.Float32, DataFormat.Float16_b),
]

_FLOAT_TO_INT = [
    (DataFormat.Float32, DataFormat.Int32),
    (DataFormat.Float32, DataFormat.UInt32),
    (DataFormat.Float32, DataFormat.UInt16),
    (DataFormat.Float32, DataFormat.UInt8),
    (DataFormat.Float16_b, DataFormat.Int32),
    (DataFormat.Float16_b, DataFormat.UInt32),
    (DataFormat.Float16_b, DataFormat.UInt16),
    (DataFormat.Float16_b, DataFormat.UInt8),
]

_INT_TO_FLOAT = [
    (DataFormat.Int32, DataFormat.Float32),
    (DataFormat.Int32, DataFormat.Float16_b),
    (DataFormat.UInt32, DataFormat.Float32),
    (DataFormat.UInt32, DataFormat.Float16_b),
    (DataFormat.UInt16, DataFormat.Float32),
    (DataFormat.UInt16, DataFormat.Float16_b),
    (DataFormat.UInt8, DataFormat.Float32),
    (DataFormat.UInt8, DataFormat.Float16_b),
]

_INT_TO_INT = [
    (DataFormat.UInt16, DataFormat.UInt32),
    (DataFormat.UInt16, DataFormat.Int32),
    (DataFormat.UInt32, DataFormat.UInt16),
    (DataFormat.Int32, DataFormat.UInt16),
    (DataFormat.UInt32, DataFormat.UInt8),
    (DataFormat.Int32, DataFormat.UInt8),
    (DataFormat.UInt16, DataFormat.UInt8),
    (DataFormat.UInt8, DataFormat.UInt16),
]

TYPECAST_PAIRS = [
    InputOutputFormat(in_fmt, out_fmt)
    for in_fmt, out_fmt in (
        _FLOAT_TO_FLOAT + _FLOAT_TO_INT + _INT_TO_FLOAT + _INT_TO_INT
    )
]


def _whole_number_float_spec() -> StimuliSpec:
    """Float stimuli restricted to whole numbers in [0, 200].

    Keeps float->int truncation unambiguous (trunc == round, so this also
    matches the rounding kernels like fp32->uint16) and stays within the range
    that bfloat16 represents exactly, so fp32->fp16b is lossless in the golden.
    """

    def dist(size, dtype, generator):
        return torch.randint(0, 201, (size,), generator=generator).to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


def _required_dest_acc(formats: InputOutputFormat) -> DestAccumulation:
    """32-bit dest is mandatory whenever a 32-bit format is involved."""
    if formats.input_format.is_32_bit() or formats.output_format.is_32_bit():
        return DestAccumulation.Yes
    return DestAccumulation.No


@parametrize(
    formats=TYPECAST_PAIRS,
    approx_mode=[ApproximationMode.No],
    input_dimensions=[[64, 64], [32, 256]],
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

    # Whole-number floats for float inputs (unambiguous truncation, lossless
    # bf16); integer inputs keep their format-aware default spec.
    spec_A = None if formats.input_format.is_integer() else _whole_number_float_spec()

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
