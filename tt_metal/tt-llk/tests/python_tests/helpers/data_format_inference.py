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
    DataFormat.MxFp4_2x_A,
    DataFormat.MxFp4_2x_B,
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


_SRCAB_ONLY_FORMATS = {
    DataFormat.MxFp4_2x_A: ChipArchitecture.QUASAR,
    DataFormat.MxFp4_2x_B: ChipArchitecture.QUASAR,
}


def infer_unpack_out(
    input_format: DataFormat,
    output_format: DataFormat,
    is_fp32_dest_acc_en: DestAccumulation,
    unpacking_to_dest: bool = False,
    unpacking_to_srcs: bool = False,
    register_format_hint: Optional[DataFormat] = None,
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
        register_format_hint: Optional opt-in for a SrcA/SrcB-only register format
            (MxFp4_2x_A or MxFp4_2x_B). When set, returned as the unpack-out format
            instead of the default. Only valid when input_format == MxFp4; the hint is
            compatible with any output_format (output_format is not constrained here).
            The exponent family it implies for downstream math/pack is derived in
            infer_data_formats via infer_downstream_unpack_out (2x_A -> Float16, 2x_B -> Float16_b).

    Returns:
        The inferred output data format for unpacking to registers
    """
    # 2x-packed SrcA/SrcB opt-in: caller explicitly requests MxFp4 to be stored
    # in src registers as MxFp4_2x_A / MxFp4_2x_B (vs. default unpack-to-Float16_b path).
    if register_format_hint is not None:
        if unpacking_to_dest:
            raise ValueError(
                f"register_format_hint={register_format_hint.name} is a SrcA/SrcB-only register "
                "format and cannot be used when unpacking_to_dest=True. The 2x-packed formats "
                "are not valid Dest register formats."
            )
        if register_format_hint not in _SRCAB_ONLY_FORMATS:
            raise ValueError(
                f"register_format_hint={register_format_hint.name} is not a supported "
                f"SrcA/SrcB-only register format."
            )
        if _SRCAB_ONLY_FORMATS[register_format_hint] != get_chip_architecture():
            raise ValueError(
                f"{register_format_hint.name} is only valid on "
                f"{_SRCAB_ONLY_FORMATS[register_format_hint].value}"
            )
        if input_format == DataFormat.MxFp4 and register_format_hint not in [
            DataFormat.MxFp4_2x_A,
            DataFormat.MxFp4_2x_B,
        ]:
            raise ValueError(
                f"register_format_hint={register_format_hint.name} is not compatible with input_format={input_format.name}."
            )
        return register_format_hint

    # MX formats can only exist in L1, not in registers. Hardware unpacks MX to bfloat16 for math.
    # it can also unpack into float16 and TF32 but bfloat16 is the default for MX inputs and default in metal in general.
    if input_format.is_mx_format():
        return DataFormat.Float16_b

    # Sub-byte BFP formats can only exist in L1. For UNPACR configuration,
    # BFP2/BFP4/BFP8 inputs keep matching InDataFormat/OutDataFormat values;
    # the unpacker internally normalizes the datum into the BF16 source-register
    # representation before math consumes it.
    if input_format in [DataFormat.Bfp4_b, DataFormat.Bfp2_b]:
        return input_format

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
    if output_format in [DataFormat.Bfp4_b, DataFormat.Bfp2_b]:
        return (
            DataFormat.Float32
            if is_fp32_dest_acc_en == DestAccumulation.Yes
            else DataFormat.Float16_b
        )

    # Default:
    # With float32 dest reg datums, packer gasket can do any conversion thus packer input can be the desired output format
    # Otherwise, packer input stays equal to the dest register format (unpack_out) and packer performs conversion instead of the packer gasket
    return output_format if is_fp32_dest_acc_en == DestAccumulation.Yes else unpack_out


def infer_downstream_unpack_out(unpack_out: DataFormat) -> DataFormat:
    # Keep BFP2/BFP4 only in the UNPACR config fields; downstream test
    # inference historically models the unpacked payload as BFP8_b.
    if unpack_out in [DataFormat.Bfp4_b, DataFormat.Bfp2_b]:
        return DataFormat.Bfp8_b

    # Map a 2x-packed SrcA/SrcB-only register format back to its paired non-2x family member,
    # used to derive a math/pack format (those fields cannot hold the 2x format itself).
    if unpack_out == DataFormat.MxFp4_2x_A:
        return DataFormat.Float16
    if unpack_out == DataFormat.MxFp4_2x_B:
        return DataFormat.Float16_b
    return unpack_out


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
    register_format_hint: Optional[DataFormat] = None,
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
        unpacking_to_srcs: Whether unpacking also targets SrcS (default: False). When True, the SrcS unpack-out is inferred via a separate path.
        register_format_hint: Optional opt-in SrcA/SrcB-only register format (e.g. MxFp4_2x_A / MxFp4_2x_B). When set,
        overrides the default unpack_*_dst inferred for the input (e.g. for MxFp4 input, default is Float16_b).
        Incompatible with unpacking_to_dest=True.

    Returns:
        FormatConfig struct containing all inferred formats. The same_src_format field
        will be True when src_A and src_B use the same input format (i.e., when
        input_format_B is not provided or equals input_format), and False otherwise.
    """

    if chip_arch is None:
        chip_arch = get_chip_architecture()

    # On Quasar the math and SFPU data formats can differ. Quasar has only one 16-bit integer HW
    # encoding, Int16 -- the unpacker, the SrcA/SrcB/dest register files, and the packer all lack a
    # UInt16 encoding, so UInt16 is pass-through as Int16 across the whole unpack/math/pack datapath.
    # The unsigned-16 semantics exist only as an SFPU access mode (sfpmem::UINT16), so we carry the
    # real UInt16 intent in sfpu_math to still test SFPU support for it.
    sfpu_math_override = None
    if chip_arch == ChipArchitecture.QUASAR:
        if input_format == DataFormat.UInt16:
            input_format = DataFormat.Int16
            sfpu_math_override = DataFormat.UInt16
        if input_format_B == DataFormat.UInt16:
            input_format_B = DataFormat.Int16
            sfpu_math_override = DataFormat.UInt16
        if output_format == DataFormat.UInt16:
            output_format = DataFormat.Int16
            sfpu_math_override = DataFormat.UInt16

    # Determine the intermediate formats
    unpack_out_A = infer_unpack_out(
        input_format,
        output_format,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        register_format_hint=register_format_hint,
    )

    # Infer unpack_out_B based on input_format_B separately.
    unpack_out_B = (
        unpack_out_A
        if input_format_B is None
        else infer_unpack_out(
            input_format_B,
            output_format,
            is_fp32_dest_acc_en,
            unpacking_to_dest,
            register_format_hint=register_format_hint,
        )
    )

    # Infer unpack_out_S using separate logic for SrcS.
    # SrcS does not support 2x packing, so the hint is intentionally not forwarded here.
    unpack_out_S = infer_unpack_out(
        input_format,
        output_format,
        is_fp32_dest_acc_en,
        unpacking_to_dest,
        unpacking_to_srcs,
    )

    downstream_unpack_out_A = infer_downstream_unpack_out(unpack_out_A)
    downstream_unpack_out_B = infer_downstream_unpack_out(unpack_out_B)
    downstream_unpack_out_S = infer_downstream_unpack_out(unpack_out_S)

    # The data format used for mathematical computations, desired format in dest register (typically matches unpack_out if both regs have same format)
    math = infer_math_format(downstream_unpack_out_A, downstream_unpack_out_B)

    # FP8 is a compressed L1 format; hardware unpacks it to Float16 (float16_a) in
    # source registers. The ALU and packer must see Float16, not Lf8/Fp8_e4m3.
    if math == DataFormat.Fp8_e4m3:
        math = DataFormat.Float16

    # SFPU-side math format: same as math unless the SFPU operates in a format with no Tensix HW
    # encoding (UInt16 on Quasar), in which case the unpack/math/pack datapath was defaulted to Int16
    # above and only sfpu_math retains the UInt16 intent (via the sfpmem::UINT16 SFPU access mode).
    sfpu_math = sfpu_math_override if sfpu_math_override is not None else math

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
        downstream_unpack_out_S,
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
        sfpu_math=sfpu_math,
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
    register_format_hint: Optional[DataFormat] = None,
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
                                  Used for testing specific math kernels with explicit format requirements. Incompatible with `register_format_hint`
                                  (which is an inference-time directive); passing both raises ValueError.
        input_format_B: Optional input data format for src_B if different from src_A, used for testing specific scenarios with different A and B formats.
        unpacking_to_srcs: Whether the unpacker target is SrcS (default: False)
        register_format_hint: Optional opt-in for a SrcA/SrcB-only register format (e.g. MxFp4_2x_A / MxFp4_2x_B). When set, the inferred
                              unpack_A_dst / unpack_B_dst become the hint instead of the default (e.g. for MxFp4 input, Float16_b). Honored
                              by the inference path only; must be paired with `disable_format_inference=False`. Currently valid only when
                              `input_format == DataFormat.MxFp4`; the hint is compatible with any `output_format`. The exponent family it
                              implies for downstream math/pack is derived via infer_downstream_unpack_out (2x_A -> Float16, 2x_B -> Float16_b).
    Returns:
        A list of FormatConfig objects of length num_iterations
    """

    if disable_format_inference:
        if register_format_hint is not None:
            raise ValueError(
                f"register_format_hint={register_format_hint.name} cannot be used with "
                "disable_format_inference=True. The hint is honored by the inference path; "
                "disabling inference would silently ignore it. Either drop the hint, or "
                "set disable_format_inference=False so inference picks up the hint."
            )

        # MX formats can't exist in registers, so "keep formats the same in dest"
        # is meaningless for them. Delegate to the inference path, which produces
        # the only valid config (unpack_dst=Float16_b, math=Float16_b, etc.).
        if input_format.is_mx_format() or (
            input_format_B is not None and input_format_B.is_mx_format()
        ):
            unpack_dst = DataFormat.Float16_b
            math_format = DataFormat.Float16_b
            # When dest_acc is enabled (FP32 destination), pack_src should be Float32 to match hardware behavior
            # This affects ReLU threshold encoding - FP32 dest requires threshold in different position
            pack_src_format = (
                DataFormat.Float32
                if is_fp32_dest_acc_en == DestAccumulation.Yes
                else DataFormat.Float16_b
            )
        elif input_format == DataFormat.Fp8_e4m3:
            unpack_dst = DataFormat.Fp8_e4m3
            math_format = DataFormat.Float16
            pack_src_format = DataFormat.Float16
        else:
            unpack_dst = input_format
            math_format = input_format
            pack_src_format = input_format
            # Widening reductions (e.g. UInt16 reduce-sum) keep a narrow input but accumulate into a
            # wider 32-bit value in a 32-bit dest. The kernel masks the garbage high bits on load (driven
            # by the narrow math format) yet stores the full 32-bit result, so the packer must read the
            # whole dest word: set pack_src to the 32-bit output format while leaving unpack/math narrow.
            if (
                is_fp32_dest_acc_en == DestAccumulation.Yes
                and not input_format.is_32_bit()
                and output_format.is_32_bit()
            ):
                pack_src_format = output_format

        # Even with inference disabled, UInt16 must ride the Int16 data path on Quasar: there is no
        # UInt16 HW encoding in the unpacker, the register files/dest, or the packer, so the whole
        # datapath defaults to Int16 and only sfpu_math keeps the UInt16 intent. See infer_data_formats.
        sfpu_math_format = math_format
        resolved_arch = chip_arch if chip_arch is not None else get_chip_architecture()
        if (
            resolved_arch == ChipArchitecture.QUASAR
            and input_format == DataFormat.UInt16
        ):
            unpack_dst = DataFormat.Int16
            math_format = DataFormat.Int16
            pack_src_format = DataFormat.Int16
            sfpu_math_format = DataFormat.UInt16
            input_format = DataFormat.Int16
            output_format = (
                DataFormat.Int16
                if output_format == DataFormat.UInt16
                else output_format
            )
            if input_format_B == DataFormat.UInt16:
                input_format_B = DataFormat.Int16

        same_src_format = (input_format_B is None) or (input_format_B == input_format)
        unpack_B_src_val = (
            input_format_B if input_format_B is not None else input_format
        )

        if input_format_B is not None and input_format_B == DataFormat.Fp8_e4m3:
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
                sfpu_math=sfpu_math_format,
                same_src_format=same_src_format,
                unpack_B_src=unpack_B_src_val,
                unpack_B_dst=unpack_B_dst_val,
                unpack_S_src=input_format,
                unpack_S_dst=unpack_dst,
                pack_S_src=pack_src_format,
                pack_S_dst=output_format,
            )
        ]

    if num_iterations > 1:
        intermediate_config = infer_data_formats(
            input_format,
            input_format,
            is_fp32_dest_acc_en,
            unpacking_to_dest,
            chip_arch,
            input_format_B,
            unpacking_to_srcs,
            register_format_hint=register_format_hint,
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
        register_format_hint=register_format_hint,
    )

    return build_data_formats(num_iterations, intermediate_config, final_config)
