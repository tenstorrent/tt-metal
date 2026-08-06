# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared TRISC3 isolate-SFPU boilerplate for Quasar.

Holds the two cross-cutting concerns the per-op units do not own: the
MATH<->TRISC3 dest handshake (semaphore::FPU_SFPU / SFPU_FPU) used by
dest-path isolate nodes, and the SrcS buffer-descriptor emission used by
self-contained ones.
"""

from typing import TYPE_CHECKING, List

from helpers.llk_params import DestSync

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation
    from fuser.operand import Operand


def emit_operand_init(operands: List["Operand"]) -> str:
    """Declare and register the SrcS buffer descriptors owned by this thread.

    Mirrors OperandRegistry.emit_operand_init for the UNPACK/PACK threads, but
    emits SrcS-shaped descriptors (x=XDIM, y=ydim(mode), z=ZDIM) rather than
    tile-shaped ones: UNP_S and PACK1 walk a tile as slice_count slices, not as
    faces, so construct_tdma_desc's TensorShape geometry does not apply.
    """
    code = ""
    for operand in operands:
        code += operand.cpp_srcs_tdma_decl_init()
        code += operand.emit_srcs_buf_desc_table_entry()
    return code


def sfpu_math_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    num_sem = 2 if operation.dest_sync == DestSync.Half else 1
    return (
        f"// Wait for isolate-SFPU to consume prior op's FPU handoffs\n"
        f"while (semaphore_read(semaphore::FPU_SFPU) > 0) {{}};\n"
        f"// Absorb isolate-SFPU's leftover dest-done posts before reseeding\n"
        f"while (semaphore_read(semaphore::SFPU_FPU) > 0)\n"
        f"{{\n"
        f"    _llk_sync_get_<>(semaphore::SFPU_FPU);\n"
        f"}};\n"
        f"_reset_dest_register_offset_();\n"
        f"_set_dest_section_base_<TRISC_ID>(_get_dest_buffer_base_());\n"
        f"_llk_sync_init_(semaphore::FPU_SFPU, {num_sem}, 0);\n"
        f"_llk_sync_init_(semaphore::SFPU_FPU, {num_sem}, 0);\n"
    )


def math_signal_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return (
        f"_llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, "
        f"p_stall::STALL_ON_MAX>(semaphore::FPU_SFPU);\n"
        f"_llk_sync_post_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);\n"
    )


def math_wait_for_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return (
        f"_llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, "
        f"p_stall::STALL_ON_ZERO>(semaphore::SFPU_FPU);\n"
        f"_llk_sync_get_<>(semaphore::SFPU_FPU);\n"
    )


def sfpu_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    return (
        f"_reset_dest_register_offset_();\n"
        f"_set_dest_section_base_<TRISC_ID>(_get_dest_buffer_base_());\n"
    )


def sfpu_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return (
        f"_llk_sync_wait_<p_stall::STALL_SFPU | p_stall::STALL_SYNC, "
        f"p_stall::STALL_ON_ZERO>(semaphore::FPU_SFPU);\n"
        f"_llk_sync_get_<>(semaphore::FPU_SFPU);\n"
    )


def sfpu_signal_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return (
        f"_llk_sync_wait_<p_stall::STALL_SFPU | p_stall::STALL_SYNC, "
        f"p_stall::STALL_ON_MAX>(semaphore::SFPU_FPU);\n"
        f"_llk_sync_post_<p_stall::WAIT_SFPU>(semaphore::SFPU_FPU);\n"
    )


def sfpu_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    if operation.dest_sync != DestSync.Half:
        return ""
    en_32bit_dest = config.dest_acc.cpp_enum_value
    return f"_llk_sync_advance_dest_section_<TRISC_ID, {en_32bit_dest}, p_stall::WAIT_SFPU>();\n"
