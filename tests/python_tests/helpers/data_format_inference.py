# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Data format inference for LLK pipeline stages.

This module provides functionality to automatically infer data formats across
the unpacking, math, and packing stages of compute pipelines, handling
architecture-specific differences between Wormhole and Blackhole.
"""
from typing import List, Optional

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .format_config import DataFormat, FormatConfig
from .llk_params import DestAccumulation


def is_format_combination_outlier(
    input_format: DataFormat,
    output_format: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
) -> bool:
    """
    Checks if the given input/output format combination is an outlier case
    that is unsupported by hardware and requires a workaround.

    This outlier case occurs when converting an 8-bit exponent format datum
    directly to Float16 without using an intermediate Float32 representation
    in the dest register.

    To handle this hardware limitation, the destination register stores 32-bit datums,
    and the packer input format is converted to Float32.

    Args:
        input_format: The input data format in L1
        output_format: The output data format in L1
        is_fp32_dest_acc_en: Flag indicating if 32-bit destination accumulation is enabled (dest_acc)

    Returns:
        True if the format combination is an unsupported hardware outlier; False otherwise
    """
    return (
        input_format.is_exponent_B()
        and not input_format.is_float32()
        and output_format == DataFormat.Float16
        and is_fp32_dest_acc_en == DestAccumulation.No
    )


def infer_unpack_out(
    input_format: DataFormat,
    output_format: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    unpacking_to_dest: bool = False,
) -> DataFormat:
    """
    Returns the output format for the unpacker (data format config for registers)
    based on the input format in L1 and whether unpacking targets the source or destination register.

    Note:
        For Quasar, the unpacker can perform data conversions, but for now only conversions performed by the packer are tested.
        For Quasar, the conditions determining which format Float32 truncates to are designed to minimize exponent mixing, but are not a hardware limitation.

    Args:
        input_format: The data format currently stored in L1 cache
        output_format: The final desired output format
        is_fp32_dest_acc_en: Whether FP32 accumulation is enabled
        unpacking_to_dest: Indicates whether unpacking targets the destination register

    Returns:
        The inferred output data format for unpacking to registers
    """
    if input_format == DataFormat.Float32 and not unpacking_to_dest:
        # When input format in L1 is Float32 + unpacking to src registers (instead of directly to dest register)
        # Source registers can store 19-bit values, so we truncate Float32 to Tf32 if we know dest will be 32-bit format
        # which preserves the 8-bit exponent and as much mantissa precision as fits. If our dst register is 16-bit we directly truncate to 16-bit format
        if is_fp32_dest_acc_en == DestAccumulation.Yes:
            return DataFormat.Tf32
        elif output_format.is_exponent_B():  # includes Float32
            return DataFormat.Float16_b  # If output Float32 or Float16_b
        return DataFormat.Float16  # Tilize to Float16

    # For all other cases, we can keep the format the same in L1 and src register or dest register
    return input_format


def infer_pack_in(
    input_format: DataFormat,
    output_format: DataFormat,
    unpack_out: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    unpacking_to_dest: bool = False,
    chip_arch: Optional[ChipArchitecture] = None,
) -> DataFormat:
    """
    Infers the packer input format based on input/output formats and architecture.

    Args:
        input_format: Input data format in L1 (unpacker input)
        output_format: Final output data format after packing
        unpack_out: The unpacker output format
        is_fp32_dest_acc_en: Flag indicating if FP32 accumulation is enabled
        unpacking_to_dest: Whether unpacking targets the destination register
        chip_arch: The chip architecture (Wormhole or Blackhole). If None, will be detected automatically.

    Returns:
        The inferred packer input format
    """
    if chip_arch is None:
        chip_arch = get_chip_architecture()

    is_wormhole = chip_arch == ChipArchitecture.WORMHOLE
    is_quasar = chip_arch == ChipArchitecture.QUASAR

    if is_quasar:
        if (
            input_format in (DataFormat.Float16, DataFormat.Float16_b)
            and output_format == DataFormat.Float32
            and is_fp32_dest_acc_en == DestAccumulation.No
        ):
            # When the dest register is in 32-bit mode, input_fmt=Fp16/16_b -> output_fmt=Fp32 is valid
            # because pack_in=pack_out=Fp32, which is a supported packer conversion.
            # When dest register is in 16-bit mode, input_fmt=Fp16/16_b -> output_fmt=Fp32 is not valid
            # because pack_in=Fp16/16_b and pack_out=Fp32, which is not a supported packer conversion.
            raise ValueError(
                "Quasar packer does not support {input_format.name} to Float32 conversion when the dest register is in 16-bit mode"
            )
        # When the dest register is in 32-bit mode, the packer input format is 32-bit
        return (
            DataFormat.Float32
            if is_fp32_dest_acc_en == DestAccumulation.Yes
            else input_format
        )

    # Wormhole + FP32 dest reg datums + Float16 output: keep Float32 for packer input for conversion to desired output format
    if (
        is_wormhole
        and is_fp32_dest_acc_en == DestAccumulation.Yes
        and output_format == DataFormat.Float16
    ):
        # On wormhole architecture, datums stored as Float32 in dest register,
        # gasket cannot convert Float32 ->Float16_A, so the packer must do the conversion,
        # we leave float32 datums in dest register allowing the packer to handle the conversion successfully.
        return DataFormat.Float32

    # Float32 in L1, unpacking to src regs: choose directly if packer can convert
    if input_format == DataFormat.Float32 and not unpacking_to_dest:
        if (
            is_fp32_dest_acc_en == DestAccumulation.Yes
            or output_format.is_exponent_B()
            and not output_format.is_float32()
        ):
            # If float32 dest reg datums and the output format has an 8-bit exponent,
            # the packer input format can directly be the output format since packer can convert Float32 to another 8-bit exponent format
            return output_format
        # Otherwise use the unpacker output (Tf32 or 16-bit) as packer input
        return unpack_out

    # Float16_A in L1 to Bfp8_B without float32 datums in dest reg requires Bfp8_A as packer input for conversion to desired output format
    if (
        input_format == DataFormat.Float16
        and output_format == DataFormat.Bfp8_b
        and is_fp32_dest_acc_en == DestAccumulation.No
    ):
        return DataFormat.Bfp8

    # 8-bit exponent -> Float16 without float32 datums in dest reg requires Float32 on Wormhole
    elif is_format_combination_outlier(
        input_format, output_format, is_fp32_dest_acc_en
    ):
        # Handling a hardware limitation: cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register.
        # For wormhole architecture, gasket cannot perform this conversion and packer takes input Float32 (from dest register) converting to Float16_A.
        # For blackhole architecture, gasket able to convert Float32 to Float16_A before packing (reduces work on packer).
        return DataFormat.Float32 if is_wormhole else output_format

    # Default:
    # With float32 dest reg datums, packer gasket can do any conversion thus packer input can be the desired output format
    # Otherwise, packer input stays equal to the dest register format (input_format)and packer performs conversion instead of the packer gasket
    return (
        output_format if is_fp32_dest_acc_en == DestAccumulation.Yes else input_format
    )


def infer_data_formats(
    input_format: DataFormat,
    output_format: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    unpacking_to_dest: bool = False,
    chip_arch: Optional[ChipArchitecture] = None,
) -> FormatConfig:
    """
    Infers all data formats needed for unpacking, math, and packing stages in a pipeline.

    Args:
        input_format: Input data format in L1 (unpacker input)
        output_format: Final output data format after packing
        is_fp32_dest_acc_en: Flag indicating if FP32 accumulation is enabled
        unpacking_to_dest: Whether unpacking targets the destination register (default: False)
        chip_arch: The chip architecture (Wormhole or Blackhole). If None, will be detected automatically.

    Returns:
        FormatConfig struct containing all formats (with same_src_format=True, so A and B formats match)
    """
    if chip_arch is None:
        chip_arch = get_chip_architecture()

    # Determine the intermediate formats
    unpack_out = infer_unpack_out(
        input_format, output_format, is_fp32_dest_acc_en, unpacking_to_dest
    )  # output format for Unpacker, desired format in src register(s)
    math = unpack_out  # The data format used for mathematical computations, desired format in dest register (typically matches unpack_out)
    pack_in = infer_pack_in(
        input_format,
        output_format,
        unpack_out,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        chip_arch,
    )  # input to the packing stage, determines what gasket can convert from dest register
    # potentially different from unpack_out and pack_out depending on FP32 accumulation

    # Return a FormatConfig struct capturing all the inferred formats needed for this stage
    # same_src_format=True by default, so unpack_B_src/dst will match unpack_A_src/dst
    return FormatConfig(
        unpack_A_src=input_format,
        unpack_A_dst=unpack_out,
        math=math,
        pack_src=pack_in,
        pack_dst=output_format,
    )


def build_data_formats(
    num_iterations: int,
    intermediate_config: FormatConfig,
    final_config: FormatConfig,
) -> List[FormatConfig]:
    """
    Helper function to build a list of FormatConfig objects.

    This function generates a list of FormatConfig, simulating
    multiple pipeline iterations where all but the last use intermediate_config,
    and the last uses final_config.

    Args:
        num_iterations: Number of L1-to-L1 pipeline iterations (list size)
        intermediate_config: Configuration for all iterations except the last
        final_config: Configuration for the final iteration

    Returns:
        List of FormatConfig for each iteration
    """

    return (
        [intermediate_config] * (num_iterations - 1) + [final_config]
        if num_iterations > 0
        else []
    )


def data_formats(
    input_format: DataFormat,
    output_format: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    num_iterations: int,
    unpacking_to_dest: bool = False,
    chip_arch: Optional[ChipArchitecture] = None,
    disable_format_inference: bool = False,
) -> List[FormatConfig]:
    """
    Entry point for computing a list of FormatConfig objects.

    Each FormatConfig object contains all the data formats necessary to execute
    a specific L1-to-L1 compute run across all 3 cores: unpack, math, and pack.

    Args:
        input_format: The input data format for all pipeline runs
        output_format: The output data format for the final pipeline run
        is_fp32_dest_acc_en: Whether FP32 accumulation is enabled
        num_iterations: The number of pipeline runs (iterations), determines list length
        unpacking_to_dest: Whether unpacking targets the destination register (default: False)
        chip_arch: The chip architecture (Wormhole or Blackhole). If None, will be detected automatically.
        disable_format_inference: When True, disables automatic data format inference and conversions, ensuring input formats are the same in dest.
                                  Used for testing specific math kernels with explicit format requirements.

    Returns:
        A list of FormatConfig objects of length num_iterations
    """

    if chip_arch is None:
        chip_arch = get_chip_architecture()

    if disable_format_inference:
        # Return a single FormatConfig where all formats are the same if format inference is disabled or not supported for the architecture
        return [
            FormatConfig(
                unpack_A_src=input_format,
                unpack_A_dst=input_format,
                math=input_format,
                pack_src=input_format,
                pack_dst=output_format,
            )
        ]  # No final config for single iteration

    intermediate_config = infer_data_formats(
        input_format, input_format, is_fp32_dest_acc_en, unpacking_to_dest, chip_arch
    )
    final_config = infer_data_formats(
        input_format, output_format, is_fp32_dest_acc_en, unpacking_to_dest, chip_arch
    )

    return build_data_formats(num_iterations, intermediate_config, final_config)
