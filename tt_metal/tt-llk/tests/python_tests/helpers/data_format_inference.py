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

VALID_QUASAR_SRC_REG_FORMATS = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
    DataFormat.Tf32,
    DataFormat.Int32,
    DataFormat.Int8,
    DataFormat.UInt8,
    DataFormat.Int16,
]

VALID_QUASAR_DEST_REG_FORMATS = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
    DataFormat.Int32,
    DataFormat.Int8,
    DataFormat.UInt8,
    DataFormat.Int16,
]


def validate_quasar_data_formats(
    unpack_out_A: DataFormat,
    unpack_out_B: DataFormat,
    pack_in: DataFormat,
    unpacking_to_dest: bool,
):
    """
    Checks if the given unpack_out and pack_in formats are supported as source or destination register formats on the Quasar architecture.

    Args:
        unpack_out_A: The unpack_out_A data format, also the source register A format, or destination register format if unpacking to dest
        unpack_out_B: The unpack_out_B data format, also the source register B format
        pack_in: The pack_in data format, also the destination register format
        unpacking_to_dest: Flag indicating if unpacking targets the destination register

    Returns:
        None if the formats are valid; raises a ValueError otherwise
    """
    if unpacking_to_dest:
        if unpack_out_A not in VALID_QUASAR_DEST_REG_FORMATS:
            raise ValueError(
                f"Unpack_out_A format {unpack_out_A.name} is not a supported Dest register format"
            )
    else:
        if unpack_out_A not in VALID_QUASAR_SRC_REG_FORMATS:
            raise ValueError(
                f"Unpack_out_A format {unpack_out_A.name} is not a supported SrcA register format"
            )
        if unpack_out_B not in VALID_QUASAR_SRC_REG_FORMATS:
            raise ValueError(
                f"Unpack_out_B format {unpack_out_B.name} is not a supported SrcB register format"
            )
        if pack_in not in VALID_QUASAR_DEST_REG_FORMATS:
            raise ValueError(
                f"Pack_in format {pack_in.name} is not a supported Dest register format"
            )


def _check_register_format(data_format: DataFormat, valid_formats: list, role: str):
    if data_format not in valid_formats:
        raise ValueError(f"Inferred {role} format {data_format.name} is not supported")


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
    unpacking_to_srcs: bool = False,
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
    # MX formats can only exist in L1, not in registers. Hardware unpacks MX to bfloat16 for math.
    if input_format.is_mx_format():
        return DataFormat.Float16_b

    # Sub-byte BFP formats (Bfp4_b) can only exist in L1.
    # The Wormhole HW unpacker only supports BFP4_b → BFP8_b conversion
    # (not BFP4_b → Float16_b). The unpacker expands 4-bit mantissas to 8-bit
    # in the BFP8_b register format, preserving the shared exponent structure.
    if input_format == DataFormat.Bfp4_b:
        return DataFormat.Bfp8_b

    if (
        input_format == DataFormat.Float32
        and not unpacking_to_dest
        and not unpacking_to_srcs
    ):
        # When input format in L1 is Float32 + unpacking to src registers (instead of directly to dest register)
        # Source registers can store 19-bit values, so we truncate Float32 to Tf32 if we know dest will be 32-bit format
        # which preserves the 8-bit exponent and as much mantissa precision as fits. If our dst register is 16-bit we directly truncate to 16-bit format
        if is_fp32_dest_acc_en == DestAccumulation.Yes:
            return DataFormat.Tf32
        elif output_format.is_exponent_B():  # includes Float32
            return DataFormat.Float16_b  # If output Float32 or Float16_b
        return DataFormat.Float16  # Tilize to Float16

    if unpacking_to_srcs and is_fp32_dest_acc_en == DestAccumulation.Yes:
        return DataFormat.Float32

    # For all other cases, we can keep the format the same in L1 and src register or dest register
    return input_format


def infer_pack_in(
    input_format: DataFormat,  # Parameter not used but kept for future use.
    output_format: DataFormat,
    unpack_out: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    unpacking_to_dest: bool = False,
    chip_arch: Optional[ChipArchitecture] = None,
    unpacking_to_srcs: bool = False,
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

    # Packer operates on register data, so use unpack_out (what's in registers) not input_format (what's in L1).
    # For MX formats, unpack_out is already Float16_b (handled in infer_unpack_out).

    if is_quasar:
        if output_format.is_32_bit() and is_fp32_dest_acc_en == DestAccumulation.No:
            # When the dest register is in 32-bit mode, input_fmt=Fp16/16_b -> output_fmt=Fp32 is valid
            # because pack_in=pack_out=Fp32, which is a supported packer conversion.
            # When dest register is in 16-bit mode, input_fmt=Fp16/16_b -> output_fmt=Fp32 is not valid
            # because pack_in=Fp16/16_b and pack_out=Fp32, which is not a supported packer conversion.
            # Similarly, input_fmt=Int8/UInt8 -> output_fmt=Int32 is not valid when the dest register is in 16-bit mode.
            raise ValueError(
                f"Quasar packer does not support {input_format.name} to {output_format.name} conversion when the dest register is in 16-bit mode"
            )

        if unpack_out.is_integer():
            if (
                unpack_out == DataFormat.Int16
                and is_fp32_dest_acc_en == DestAccumulation.Yes
            ):
                raise ValueError(
                    f"If the input format is Int16, 32-bit dest is not supported and the packer input format must be Int16"
                )
            # When the dest register is in 32-bit mode, the packer input format is 32-bit
            return (
                DataFormat.Int32
                if is_fp32_dest_acc_en == DestAccumulation.Yes
                else unpack_out
            )
        else:
            # When the dest register is in 32-bit mode, the packer input format is 32-bit
            return (
                DataFormat.Float32
                if is_fp32_dest_acc_en == DestAccumulation.Yes
                else unpack_out
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
    if unpack_out == DataFormat.Float32 and not unpacking_to_dest:
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
        unpack_out == DataFormat.Float16
        and output_format == DataFormat.Bfp8_b
        and is_fp32_dest_acc_en == DestAccumulation.No
    ):
        return DataFormat.Bfp8

    # 8-bit exponent -> Float16 without float32 datums in dest reg requires Float32 on Wormhole
    elif is_format_combination_outlier(unpack_out, output_format, is_fp32_dest_acc_en):
        # Handling a hardware limitation: cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register.
        # For wormhole architecture, gasket cannot perform this conversion and packer takes input Float32 (from dest register) converting to Float16_A.
        # For blackhole architecture, gasket able to convert Float32 to Float16_A before packing (reduces work on packer).
        return DataFormat.Float32 if is_wormhole else output_format

    # Sub-byte BFP formats cannot be used as pack_src (packer in_data_format).
    # The packer reads 16-bit (or 32-bit with dest_acc) data from dest and converts to BFP for L1.
    if output_format == DataFormat.Bfp4_b:
        return (
            DataFormat.Float32
            if is_fp32_dest_acc_en == DestAccumulation.Yes
            else DataFormat.Float16_b
        )

    # Default:
    # With float32 dest reg datums, packer gasket can do any conversion thus packer input can be the desired output format
    # Otherwise, packer input stays equal to the dest register format (unpack_out) and packer performs conversion instead of the packer gasket
    return output_format if is_fp32_dest_acc_en == DestAccumulation.Yes else unpack_out


def infer_math_format(a: DataFormat, b: DataFormat = None) -> DataFormat:
    # Design rationale:
    # - The only constraint enforced here is that both inputs belong to the same exponent "family"
    #   (e.g. exponent-B vs. non-exponent-B). Mixing families is not supported and is rejected above.
    # - Once formats are in the same family, the math pipeline can operate in either format; any
    #   required conversions from L1 to the math format and back are already handled by unpack/pack.
    # - Therefore, choosing `a` as the math format is always correct when the compatibility check
    #   passes, and keeps the behavior deterministic and simple compared to the previous heuristic.

    if b is None:
        return a

    if (a.is_exponent_B() and not b.is_exponent_B()) or (
        not a.is_exponent_B() and b.is_exponent_B()
    ):
        raise ValueError(
            f"Incompatible formats {a.name} and {b.name} with different exponent bit widths were provided."
        )

    return a


def infer_data_formats(
    input_format: DataFormat,
    output_format: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    unpacking_to_dest: bool = False,
    chip_arch: Optional[ChipArchitecture] = None,
    input_format_B: DataFormat = None,
    unpacking_to_srcs: bool = False,
) -> FormatConfig:
    """
    Infers all data formats needed for unpacking, math, and packing stages in a pipeline.

    Args:
        input_format: Input data format in L1 (unpacker input). For multiple inputs, this is the format for src_A and input_format_B is used for src_B.
        output_format: Final output data format after packing
        is_fp32_dest_acc_en: Flag indicating if FP32 accumulation is enabled
        unpacking_to_dest: Whether unpacking targets the destination register (default: False)
        chip_arch: The chip architecture (Wormhole or Blackhole). If None, will be detected automatically.
        input_format_B: Optional input data format for src_B if different from src_A, used for testing specific scenarios with different A and B formats.

    Returns:
        FormatConfig struct containing all inferred formats. The same_src_format field
        will be True when src_A and src_B use the same input format (i.e., when
        input_format_B is not provided or equals input_format), and False otherwise.
    """

    # Determine the intermediate formats
    unpack_out_A = infer_unpack_out(
        input_format, output_format, is_fp32_dest_acc_en, unpacking_to_dest
    )

    # Infer unpack_out_B based on input_format_B separately.
    unpack_out_B = (
        unpack_out_A
        if input_format_B is None
        else infer_unpack_out(
            input_format_B, output_format, is_fp32_dest_acc_en, unpacking_to_dest
        )
    )

    # Infer unpack_out_S using separate logic for SrcS
    unpack_out_S = infer_unpack_out(
        input_format,
        output_format,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        unpacking_to_srcs,
    )

    # The data format used for mathematical computations, desired format in dest register (typically matches unpack_out if both regs have same format)
    math = infer_math_format(unpack_out_A, unpack_out_B)

    # FP8 is a compressed L1 format; hardware unpacks it to Float16 (float16_a) in
    # source registers. The ALU and packer must see Float16, not Lf8/Fp8_e4m3.
    if math == DataFormat.Fp8_e4m3:
        math = DataFormat.Float16

    pack_in = infer_pack_in(
        input_format,
        output_format,
        math,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        chip_arch,
    )  # input to the packing stage, determines what gasket can convert from dest register
    # potentially different from unpack_out and pack_out depending on FP32 accumulation

    # FP8 output: gasket must produce Float16 for packer's Pac_LF8_4b_exp encode path.
    # When math is a B-format (Float16_b), use Float16_b so the packer can distinguish
    # A-format vs B-format pipelines and enable 10-bit mantissa rounding accordingly.
    if output_format == DataFormat.Fp8_e4m3:
        pack_in = DataFormat.Float16_b if math.is_exponent_B() else DataFormat.Float16

    # Input to the SrcS packer (PACK1)
    pack_in_S = infer_pack_in(
        input_format,
        output_format,
        unpack_out_S,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        chip_arch,
        unpacking_to_srcs,
    )

    # We fall back to using input_format for src_B if input_format_B is not provided, ensuring same_src_format is True in this case.
    if input_format_B is None:
        input_format_B = input_format

    same_src_format = input_format == input_format_B

    # Check if unpack_out (src or dest reg) and pack_in (dest reg) formats are valid for Quasar
    if chip_arch == ChipArchitecture.QUASAR:
        validate_quasar_data_formats(
            unpack_out_A, unpack_out_B, pack_in, unpacking_to_dest
        )

    return FormatConfig(
        unpack_A_src=input_format,
        unpack_A_dst=unpack_out_A,
        pack_src=pack_in,
        pack_dst=output_format,
        math=math,
        same_src_format=same_src_format,
        unpack_B_src=input_format_B,
        unpack_B_dst=unpack_out_B,
        unpack_S_src=input_format,
        unpack_S_dst=unpack_out_S,
        pack_S_src=pack_in_S,
        pack_S_dst=output_format,
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
    input_format_B: DataFormat = None,
    unpacking_to_srcs: bool = False,
) -> List[FormatConfig]:
    """
    Entry point for computing a list of FormatConfig objects.

    Each FormatConfig object contains all the data formats necessary to execute
    a specific L1-to-L1 compute run across all 3 cores: unpack, math, and pack.

    Args:
        input_format: The input data format for all pipeline runs. For multiple inputs, this is the format for src_A and input_format_B is used for src_B.
        output_format: The output data format for the final pipeline run
        is_fp32_dest_acc_en: Whether FP32 accumulation is enabled
        num_iterations: The number of pipeline runs (iterations), determines list length
        unpacking_to_dest: Whether unpacking targets the destination register (default: False)
        chip_arch: The chip architecture (Wormhole or Blackhole). If None, will be detected automatically.
        disable_format_inference: When True, disables automatic data format inference and conversions, ensuring input formats are the same in dest.
                                  Used for testing specific math kernels with explicit format requirements.
        input_format_B: Optional input data format for src_B if different from src_A, used for testing specific scenarios with different A and B formats.
    Returns:
        A list of FormatConfig objects of length num_iterations
    """

    if (
        disable_format_inference
    ):  # TODO: What happens here when we have two different input formats?
        # Return a single FormatConfig where all formats are the same if format inference is disabled or not supported for the architecture
        # MX formats can only exist in L1, not in registers. Hardware unpacks MX to bfloat16 for math.
        if input_format.is_mx_format():
            unpack_dst = DataFormat.Float16_b
            math_format = DataFormat.Float16_b
            pack_src_format = DataFormat.Float16_b
        elif input_format == DataFormat.Fp8_e4m3:
            unpack_dst = DataFormat.Fp8_e4m3
            math_format = DataFormat.Float16
            pack_src_format = DataFormat.Float16
        else:
            unpack_dst = input_format
            math_format = input_format
            pack_src_format = input_format

        # Determine if we have different formats for A and B
        same_src_format = (input_format_B is None) or (input_format_B == input_format)
        unpack_B_src_val = (
            input_format_B if input_format_B is not None else input_format
        )

        # For B destination format when format inference is disabled
        if input_format_B is not None and input_format_B.is_mx_format():
            unpack_B_dst_val = DataFormat.Float16_b
        elif input_format_B is not None and input_format_B == DataFormat.Fp8_e4m3:
            unpack_B_dst_val = DataFormat.Fp8_e4m3
        elif input_format_B is not None:
            unpack_B_dst_val = input_format_B
        else:
            unpack_B_dst_val = unpack_dst

        return [
            FormatConfig(
                unpack_A_src=input_format,
                unpack_A_dst=unpack_dst,
                pack_src=pack_src_format,
                pack_dst=output_format,
                math=math_format,
                same_src_format=same_src_format,
                unpack_B_src=unpack_B_src_val,
                unpack_B_dst=unpack_B_dst_val,
                unpack_S_src=input_format,
                unpack_S_dst=unpack_dst,
                pack_S_src=pack_src_format,
                pack_S_dst=output_format,
            )
        ]  # No final config for single iteration

    if num_iterations > 1:
        intermediate_config = infer_data_formats(
            input_format,
            input_format,
            is_fp32_dest_acc_en,
            unpacking_to_dest,
            chip_arch,
            input_format_B,
            unpacking_to_srcs,
        )
    else:
        intermediate_config = None

    final_config = infer_data_formats(
        input_format,
        output_format,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        chip_arch,
        input_format_B,
        unpacking_to_srcs,
    )

    return build_data_formats(num_iterations, intermediate_config, final_config)
