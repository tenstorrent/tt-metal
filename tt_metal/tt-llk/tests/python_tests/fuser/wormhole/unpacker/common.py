# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.fpu_node import FpuNode

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def hw_configure_unpack(config: "GlobalConfig", compute_node: FpuNode) -> str:
    dest_acc = config.dest_acc.cpp_enum_value
    unpack_A_src, unpack_A_dst = compute_node.unpack_A_src, compute_node.unpack_A_dst
    unpack_B_src, unpack_B_dst = compute_node.unpack_B_src, compute_node.unpack_B_dst
    operand_a, operand_b = compute_node.unpacker.physical_operands(compute_node)
    face_r_dim_a = operand_a.tile_shape.face_r_dim
    num_faces_a = operand_a.tile_shape.total_num_faces()
    tile_size_a = operand_a.tile_size
    face_r_dim_b = operand_b.tile_shape.face_r_dim
    num_faces_b = operand_b.tile_shape.total_num_faces()
    tile_size_b = operand_b.tile_size

    return (
        f"_llk_unpack_hw_configure_<{dest_acc}>(\n"
        f"    {unpack_A_src.cpp_underlying_value}, {unpack_B_src.cpp_underlying_value},\n"
        f"    {unpack_A_dst.cpp_underlying_value}, {unpack_B_dst.cpp_underlying_value},\n"
        f"    {face_r_dim_a}, {face_r_dim_b}, {num_faces_a}, {num_faces_b},\n"
        f"    {tile_size_a}, {tile_size_b}\n"
        f");\n"
    )


def configure_unpack(
    config: "GlobalConfig", compute_node: FpuNode, previous: FpuNode
) -> str:
    dest_acc = config.dest_acc.cpp_enum_value
    old_A_src, old_B_src = previous.unpack_A_src, previous.unpack_B_src
    new_A_src, new_A_dst = compute_node.unpack_A_src, compute_node.unpack_A_dst
    new_B_src, new_B_dst = compute_node.unpack_B_src, compute_node.unpack_B_dst
    srca_changed, srcb_changed, srca_tile_changed, srcb_tile_changed = (
        compute_node.unpack_changes(previous)
    )
    code = ""
    operand_a, operand_b = compute_node.unpacker.physical_operands(compute_node)

    if srca_changed:
        new_face_r_dim_a = operand_a.tile_shape.face_r_dim
        new_num_faces_a = operand_a.tile_shape.total_num_faces()

        to_from_int8 = (
            "true"
            if old_A_src.needs_int8_math_config() or new_A_src.needs_int8_math_config()
            else "false"
        )
        dim_stride = (
            "p_dim_stride_target::FACE_ROW_MAJOR"
            if srca_tile_changed
            else "p_dim_stride_target::IGNORE"
        )
        code += (
            f"_llk_unpack_reconfig_data_format_srca_impl_<{dest_acc}, {dim_stride}, {to_from_int8}>(\n"
            f"    {new_A_src.cpp_underlying_value}, {new_A_dst.cpp_underlying_value}, {operand_a.tile_size}"
        )
        if srca_tile_changed:
            code += f", {new_face_r_dim_a}, {new_num_faces_a}"
        code += "\n);\n"

    if srcb_changed:
        srcb_tile_size = operand_b.tile_size
        new_face_r_dim_b = operand_b.tile_shape.face_r_dim
        new_num_faces_b = operand_b.tile_shape.total_num_faces()

        to_from_int8 = (
            "true"
            if old_B_src.needs_int8_math_config() or new_B_src.needs_int8_math_config()
            else "false"
        )
        dim_stride = (
            "p_dim_stride_target::FACE_ROW_MAJOR"
            if srcb_tile_changed
            else "p_dim_stride_target::IGNORE"
        )
        code += (
            f"_llk_unpack_reconfig_data_format_srcb_impl_<{dest_acc}, {dim_stride}, {to_from_int8}>(\n"
            f"    {new_B_src.cpp_underlying_value}, {new_B_dst.cpp_underlying_value}, {srcb_tile_size}"
        )
        if srcb_tile_changed:
            code += f", {new_face_r_dim_b}, {new_num_faces_b}"
        code += "\n);\n"

    return code


def dvalid_init(**kwargs) -> str:
    return ""


def unpack_to_dest_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def unpack_to_dest_wait_for_dest_available(config: "GlobalConfig") -> str:
    return ""


def unpack_to_dest_section_done(
    config: "GlobalConfig", operation: "L1Operation"
) -> str:
    return ""


def sync_with_packer(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.needs_pack_sync:
        return (
            "t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);\n"
            "t6_semaphore_get<>(semaphore::PACK_DONE);\n"
        )
    return ""
