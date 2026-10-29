# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.format_config import DataFormat
from helpers.llk_params import DestSync, PerfRunType

if TYPE_CHECKING:
    from fuser.fpu_node import FpuNode
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def fidelity_source_formats(
    config: "GlobalConfig", operation: "L1Operation", compute_unit: "FpuNode"
) -> tuple[str, str]:
    output_format = operation._get_pack_nodes()[0].output.data_format
    _, src_a, _, src_b, _, _ = config.sentinel._infer_node_formats(
        config, compute_unit, output_format, operation
    )
    src_a, src_b = compute_unit.unpacker.physical_order(src_a, src_b)
    return src_a.cpp_enum_value, src_b.cpp_enum_value


def hw_configure_math(dest_acc: str, math_fmt: DataFormat) -> str:
    return (
        f"_llk_math_srcAB_hw_configure_<{dest_acc}>(\n"
        f"    {math_fmt.cpp_enum_value}, {math_fmt.cpp_enum_value}\n"
        f");\n"
    )


def configure_math(
    dest_acc: str,
    old_math: DataFormat,
    new_math: DataFormat,
) -> str:
    math_fmt = new_math.cpp_enum_value
    return (
        f"_configure_alu_formats_<{dest_acc}>(\n"
        f"    {math_fmt}, {math_fmt},\n"
        f"    _is_src_fmt_int32_dest_compatible_({math_fmt}) && {dest_acc}, DataFormat::Invalid\n"
        f");\n"
    )


def math_pack_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    dest_sync = operation.dest_sync.cpp_enum_value
    if config.quasar_use_dvalid:
        if config.perf_run_type in (None, PerfRunType.L1_TO_L1):
            return "set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"
        return "set_up_zero_dest_dvalid_handshake_for_math();\n"
    if operation.stage_id != 1:
        return ""
    code = f"_llk_math_pack_sync_init_<{dest_sync}>();\n"
    if config.uses_unpack_to_dest_sync:
        code += (
            "set_up_zero_dest_dvalid_handshake_for_math();\n"
            "_llk_sync_barrier_pass_(semaphore::UNPACK_INIT);\n"
        )
    return code


def math_wait_for_dest(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.uses_unpack_to_dest_sync:
        code = ""
        if config.uses_unpack_to_dest_pack_sync:
            code += "_llk_math_wait_for_dest_available_();\n"
        code += "_llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::UNPACK_MATH);\n"
        return code
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return "_llk_math_wait_for_dest_available_();\n"


def math_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.uses_unpack_to_dest_sync:
        if config.uses_unpack_to_dest_pack_sync:
            code = (
                "_llk_sync_post_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::MATH_PACK);\n"
                "_llk_sync_get_(semaphore::UNPACK_MATH);\n"
            )
        else:
            code = "_llk_sync_get_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::UNPACK_MATH);\n"
        if operation.dest_sync == DestSync.Half:
            code += (
                "_llk_sync_advance_dest_section_<ckernel::TRISC_ID, "
                f"{config.dest_acc.cpp_enum_value}, p_stall::WAIT_SFPU, p_stall::MATH>();\n"
            )
        return code
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    if config.quasar_use_dvalid:
        return f"_llk_math_set_dvalid_<p_cleardvalid::FPU, {dest_sync}>();\n"
    return f"_llk_math_dest_section_done_<{dest_sync}, {dest_acc}>();\n"


def math_dest_remap_config(required: bool) -> str:
    return ""
