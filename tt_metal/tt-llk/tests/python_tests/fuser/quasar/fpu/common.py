# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.llk_params import DestSync, PerfRunType

if TYPE_CHECKING:
    from fuser.fpu_node import FpuNode
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def fidelity_source_formats(compute_unit: "FpuNode") -> tuple[str, str]:
    src_a, src_b = compute_unit.unpack_A_dst, compute_unit.unpack_B_dst
    return src_a.cpp_enum_value, src_b.cpp_enum_value


def hw_configure_math(dest_acc: str, alu_formats) -> str:
    src_a, src_b = alu_formats
    return (
        f"_llk_math_srcAB_hw_configure_<{dest_acc}>(\n"
        f"    {src_a.cpp_enum_value}, {src_b.cpp_enum_value}\n"
        f");\n"
    )


def configure_math(
    dest_acc: str,
    old_formats,
    new_formats,
) -> str:
    src_a, src_b = (fmt.cpp_enum_value for fmt in new_formats)
    return (
        f"_configure_alu_formats_<{dest_acc}>(\n"
        f"    {src_a}, {src_b},\n"
        f"    _is_src_fmt_int32_dest_compatible_({src_a}) && {dest_acc}, DataFormat::Invalid\n"
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
            "_llk_sync_spin_until_posted_(semaphore::UNPACK_MATH);\n"
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
