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


def get_perf_math_operations():
    """Return the elementwise math operations covered by Quasar perf tests."""
    return [MathOperation.Elwadd, MathOperation.Elwmul]


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

    # Local import keeps param_config free to use the constraints in this module.
    from helpers.param_config import get_num_blocks_and_num_tiles_in_block

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


# Quasar conversion capabilities for L1 -> registers, SrcA -> Dest, and
# Dest -> L1. Sources: Tensix unpacker/packer conversions and Neo FPU formats.
_QUASAR_NARROW_FORMATS = frozenset(
    {
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Fp8_e4m3,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    }
)
_QUASAR_NARROW_FLOAT_DEST_FORMATS = frozenset(
    {DataFormat.Float16, DataFormat.Float16_b}
)
_QUASAR_UNPACK_TO_DEST_FORMATS = {
    # TF32 is stored in a 32-bit Float32 container in Dest.
    DataFormat.Float32: {DataFormat.Float32} | _QUASAR_NARROW_FLOAT_DEST_FORMATS,
    DataFormat.Tf32: {DataFormat.Float32} | _QUASAR_NARROW_FLOAT_DEST_FORMATS,
    **{
        input_format: _QUASAR_NARROW_FLOAT_DEST_FORMATS
        for input_format in _QUASAR_NARROW_FORMATS
    },
    DataFormat.Int32: {DataFormat.Int32},
    DataFormat.Int16: {DataFormat.Int16},
    DataFormat.Int8: {DataFormat.Int8},
    DataFormat.UInt8: {DataFormat.UInt8},
}

# SrcA/B additionally permit a TF32 result for narrow floating-point inputs.
_QUASAR_UNPACK_TO_SRCA_FORMATS = {
    **{
        input_format: output_formats
        for input_format, output_formats in _QUASAR_UNPACK_TO_DEST_FORMATS.items()
        # Int32 may be unpacked only to Dest or SrcS, not SrcA/B.
        if input_format != DataFormat.Int32
    },
    DataFormat.Float32: {DataFormat.Tf32} | _QUASAR_NARROW_FLOAT_DEST_FORMATS,
    DataFormat.Tf32: {DataFormat.Tf32} | _QUASAR_NARROW_FLOAT_DEST_FORMATS,
    **{
        input_format: {DataFormat.Tf32} | _QUASAR_NARROW_FLOAT_DEST_FORMATS
        for input_format in _QUASAR_NARROW_FORMATS
    },
    # Quasar-only 2x-packed register formats are legal only for MXFP4 -> SrcA/B.
    DataFormat.MxFp4: {
        DataFormat.Tf32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.MxFp4_2x_A,
        DataFormat.MxFp4_2x_B,
    },
}

# Effective SrcA -> Dest conversions supported by Quasar unary datacopy. The
# LLK selects MOVA2D/MOVB2D for narrow Dest and ELWADD for 32-bit Dest; test
# generation only needs to know whether the complete datacopy path is executable.
_QUASAR_FPU_DATACOPY_DEST_FORMATS = {
    DataFormat.Float16: {DataFormat.Float16, DataFormat.Float32},
    DataFormat.Float16_b: {DataFormat.Float16_b, DataFormat.Float32},
    DataFormat.Tf32: {DataFormat.Float32},
    DataFormat.Int8: {DataFormat.Int8, DataFormat.Int32},
    DataFormat.UInt8: {DataFormat.UInt8, DataFormat.Int32},
    DataFormat.Int16: {DataFormat.Int16},
}

_QUASAR_PACK_OUTPUTS = {
    DataFormat.Float32: {
        DataFormat.Float32,
        DataFormat.Tf32,
    }
    | _QUASAR_NARROW_FORMATS,
    DataFormat.Float16: _QUASAR_NARROW_FORMATS,
    DataFormat.Float16_b: _QUASAR_NARROW_FORMATS,
    DataFormat.Int32: {DataFormat.Int32, DataFormat.Int8, DataFormat.UInt8},
    DataFormat.Int16: {DataFormat.Int16},
    DataFormat.Int8: {DataFormat.Int8},
    DataFormat.UInt8: {DataFormat.UInt8},
}


def _quasar_sfpu_hardware_format(fmt: DataFormat) -> DataFormat:
    """Map an SFPU-visible logical format to its Quasar hardware encoding."""
    return DataFormat.Int16 if fmt == DataFormat.UInt16 else fmt


def _quasar_dest_hardware_format(fmt: DataFormat) -> DataFormat:
    if fmt == DataFormat.Tf32:
        return DataFormat.Float32
    return _quasar_sfpu_hardware_format(fmt)


def _quasar_effective_sfpu_format(fmt: DataFormat) -> DataFormat:
    """Return the register format exposed to SFPU after Quasar unpacking."""
    if fmt.is_mx_format():
        return DataFormat.Float16_b
    if fmt == DataFormat.Fp8_e4m3:
        return DataFormat.Float16
    if fmt == DataFormat.Tf32:
        return DataFormat.Float32
    return _quasar_sfpu_hardware_format(fmt)


def _quasar_fpu_source_format(fmt: DataFormat) -> DataFormat:
    """Return the SrcA format consumed by the FPU datacopy staging path."""
    if fmt in (DataFormat.Float32, DataFormat.Tf32):
        return DataFormat.Tf32
    if fmt.is_mx_format():
        return DataFormat.Float16_b
    if fmt == DataFormat.Fp8_e4m3:
        return DataFormat.Float16
    return _quasar_sfpu_hardware_format(fmt)


def _quasar_dest_format_matches_mode(
    dest_format: DataFormat, dest_acc: DestAccumulation
) -> bool:
    is_32_bit = dest_format in (DataFormat.Float32, DataFormat.Int32)
    return is_32_bit == (dest_acc == DestAccumulation.Yes)


def is_valid_quasar_unpack_to_dest(
    input_format: DataFormat,
    unpack_output_format: DataFormat,
    dest_acc: DestAccumulation,
    *,
    allow_narrow_in_32bit_dest: bool = False,
) -> bool:
    """Validate an L1 -> Dest unpack conversion against the Tensix table.

    Non-32bit -> 32bit data conversions are not supported by unpacker.
    That layout is accepted only for SFPU typecast, which reads the
    narrow representation and overwrites it with the converted result.
    """
    l1_input_format = _quasar_sfpu_hardware_format(input_format)
    unpack_out_format = _quasar_dest_hardware_format(unpack_output_format)

    # Check whether the unpacker supports the requested conversion.
    if unpack_out_format not in _QUASAR_UNPACK_TO_DEST_FORMATS.get(
        l1_input_format, set()
    ):
        return False

    # Check that the register format matches the configured Dest width.
    if _quasar_dest_format_matches_mode(unpack_out_format, dest_acc):
        return True

    # Typecast may temporarily hold a narrow input in a 32-bit Dest slot; SFPU
    # overwrites it with the converted 32-bit result.
    return allow_narrow_in_32bit_dest and dest_acc == DestAccumulation.Yes


def is_valid_quasar_fpu_path(
    input_format: DataFormat,
    unpack_output_format: DataFormat,
    dest_format: DataFormat,
    dest_acc: DestAccumulation,
) -> bool:
    """Validate the complete Unpack-to-SrcA + FPU datacopy -> Dest path."""
    # UInt16 uses the Int16 encoding for SFPU input/output, but Quasar does not
    # support an Int16 Src -> UInt16 Dest FPU datacopy.
    if dest_format == DataFormat.UInt16:
        return False

    # L1 input format.
    hardware_input = _quasar_sfpu_hardware_format(input_format)
    # SrcA/B format, also used as the math format.
    hardware_unpack_output = _quasar_sfpu_hardware_format(unpack_output_format)
    # Dest format, also used as the SFPU and packer input format.
    hardware_dest = _quasar_sfpu_hardware_format(dest_format)

    return (
        hardware_unpack_output
        in _QUASAR_UNPACK_TO_SRCA_FORMATS.get(hardware_input, set())
        and hardware_dest
        in _QUASAR_FPU_DATACOPY_DEST_FORMATS.get(hardware_unpack_output, set())
        and _quasar_dest_format_matches_mode(hardware_dest, dest_acc)
    )


def is_valid_quasar_packer_conversion(
    pack_input_format: DataFormat, output_format: DataFormat
) -> bool:
    """Validate a Dest -> L1 conversion against the Tensix packer table."""
    hardware_input = _quasar_sfpu_hardware_format(pack_input_format)
    hardware_output = _quasar_sfpu_hardware_format(output_format)
    return hardware_output in _QUASAR_PACK_OUTPUTS.get(hardware_input, set())
