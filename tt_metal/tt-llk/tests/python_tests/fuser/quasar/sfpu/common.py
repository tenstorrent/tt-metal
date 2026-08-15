# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.llk_params import DestSync

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation

_MATH_WAIT_STALL = "p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC"

_SFPU_WAIT_STALL = "p_stall::STALL_SFPU | p_stall::STALL_SYNC"


def sfpu_on_isolated_trisc(config: "GlobalConfig") -> bool:
    return config.quasar_isolate_sfpu


def math_handoff_to_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    return (
        "_llk_sync_post_<p_stall::MATH>(semaphore::FPU_SFPU);\n"
        f"_llk_sync_wait_<{_MATH_WAIT_STALL}, p_stall::STALL_ON_ZERO>(semaphore::SFPU_FPU);\n"
        "_llk_sync_get_<>(semaphore::SFPU_FPU);\n"
    )


def sfpu_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    return (
        f"_llk_sync_wait_<{_SFPU_WAIT_STALL}, p_stall::STALL_ON_ZERO>(semaphore::FPU_SFPU);\n"
        "_llk_sync_get_<>(semaphore::FPU_SFPU);\n"
    )


def sfpu_signal_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    return "_llk_sync_post_<p_stall::WAIT_SFPU>(semaphore::SFPU_FPU);\n"


def sfpu_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.stage_id != 1:
        return ""
    return (
        "_reset_dest_register_offset_();\n"
        "_set_dest_section_base_<ckernel::TRISC_ID>(_get_dest_buffer_base_());\n"
    )


def sfpu_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or operation.dest_sync != DestSync.Half:
        return ""
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_sync_advance_dest_section_<ckernel::TRISC_ID, {dest_acc}, p_stall::WAIT_SFPU>();\n"
