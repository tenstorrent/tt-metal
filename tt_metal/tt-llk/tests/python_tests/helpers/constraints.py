# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block


def get_valid_dest_accumulation_modes(formats):
    """
    Base constraints for Dest Accumulation modes.

    Constraints (all architectures):
    - Dest accumulation must be ENABLED for the following format combination:
        - Input format is 32bit integer format
        Reason: HW limitation, Unpacker cannot unpack 32bit integer formats into SrcA and SrcB registers

    Constraints (Wormhole/Blackhole only):
    - Dest accumulation must be ENABLED for the following format combinations:
        - Input format has B type exponent (bfp8_b, float16_b)
        - Output format is A type exponent (float16)
        Reason: HW limitation, Packer cannot convert expB to expA, so we convert it to Float32 first as intermediate. (Source???)
    - Otherwise it can be ENABLED or DISABLED

    NOTE: There are more combos that fit this rule, but aren't handled in the codebase
        So I'm not sure if they should also be handled here.

    Constraints (Quasar only):
        - 32-bit output (Float32, Int32) requires dest_acc=Yes, packer cannot perform upcasting to 32-bit formats
        - UInt8 <-> Int8 conversions require dest_acc=Yes, because packer cannot convert UInt8 <-> Int8
        - Int16 input requires dest_acc=No, packer cannot convert Int16 to and from other formats and thus
          32-bit dest register mode is not supported when working with Int16
    """
    chip_arch = get_chip_architecture()
    in_fmt, out_fmt = formats.input, formats.output

    if in_fmt in [DataFormat.Int32, DataFormat.UInt32]:
        return [DestAccumulation.Yes]

    if chip_arch == ChipArchitecture.QUASAR:
        if out_fmt.is_32_bit():
            return [DestAccumulation.Yes]
        if (in_fmt, out_fmt) in (
            (DataFormat.UInt8, DataFormat.Int8),
            (DataFormat.Int8, DataFormat.UInt8),
        ):
            return [DestAccumulation.Yes]
        if in_fmt == DataFormat.Int16:
            return [DestAccumulation.No]
    else:
        if (
            in_fmt
            in [
                DataFormat.Bfp8_b,
                DataFormat.Bfp4_b,
                DataFormat.Bfp2_b,
                DataFormat.Float16_b,
            ]
            and out_fmt == DataFormat.Float16
        ):
            return [DestAccumulation.Yes]

    return [DestAccumulation.No, DestAccumulation.Yes]


def get_valid_math_fidelities(format, operation, PERF_RUN: bool = False):
    """
    Base constraints for Math Fidelity modes.

    - Regular mode:
        - Math fidelity must be LoFi for ElwAdd and ElwSub operations
        - Otherwise it can be LoFi, HiFi2, HiFi3, HiFi4.

    - Performance mode:
        - Ignores Math fidelity settings that are higher than necessary for full precision
    """

    if operation in [MathOperation.Elwadd, MathOperation.Elwsub]:
        return [MathFidelity.LoFi]

    # HiFi2 will multiply BFP8 and BFP8_b in full precision, skip HiFi3 and HiFi4
    if PERF_RUN and format.input in [DataFormat.Bfp8_b, DataFormat.Bfp8]:
        return [MathFidelity.LoFi, MathFidelity.HiFi2]

    # todo: once support for any of these is added, add them here
    # LoFi will multiply FP8, BFP4, BFP4a, BFP2, BFP2a in full precision, skip HiFi2 and higher
    if PERF_RUN and format.input in [
        DataFormat.Fp8_e4m3,
        DataFormat.Bfp4_b,
        DataFormat.Bfp2_b,
    ]:
        return [MathFidelity.LoFi]

    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def get_valid_dest_indices(
    dest_sync: DestSync,
    dest_acc: DestAccumulation,
    formats: InputOutputFormat,
    input_dimensions: List[int],
    all_indices: bool = False,
):
    """
    Base constraint for valid destination register indices.

    By default the function only returns the lowest and highest possible indices.
    This is to limit the number of tests. Use all_indices=True force the function to return all possible indices.
    """

    # Use this function to get the number of tiles that can fit in dest.
    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    start_index = 0
    end_index = num_tiles_in_block - 1

    if all_indices:
        return list(range(start_index, end_index + 1))

    return [start_index] if start_index == end_index else [start_index, end_index]


def is_valid_data_format_conversion(fmt: InputOutputFormat) -> bool:
    """
    Base constraints for valid data format conversions. Specific operations might have additional constraints.

    Check whether a single InputOutputFormat represents a valid data format conversion.

    Constraints (all architectures):
        - Cannot convert between integer and float formats

    Constraints (Quasar only):
        - Int16 input can only output to Int16
    """
    chip_arch = get_chip_architecture()
    in_fmt, out_fmt = fmt.input_format, fmt.output_format

    if in_fmt.is_integer() ^ out_fmt.is_integer():
        return False

    if chip_arch == ChipArchitecture.QUASAR:
        if in_fmt == DataFormat.Int16 and out_fmt != DataFormat.Int16:
            return False

    return True


def get_valid_data_format_conversions(
    formats_list: List[InputOutputFormat],
) -> List[InputOutputFormat]:
    """
    Filter a list of InputOutputFormat to only valid data format conversions.

    These are basic constraints. Specific operations might have additional constraints.
    """
    return [fmt for fmt in formats_list if is_valid_data_format_conversion(fmt)]
