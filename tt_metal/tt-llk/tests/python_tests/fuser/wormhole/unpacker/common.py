# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fpu_node import FpuNode
from helpers.format_config import DataFormat
from helpers.llk_params import EltwiseBinaryReuseDestType


def is_datacopy_node(compute_node: FpuNode) -> bool:
    return False


def is_unary_unpacker(compute_node: FpuNode) -> bool:
    return False


def hw_configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_src: DataFormat,
    unpack_A_dst: DataFormat,
    unpack_B_src: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    face_r_dim_a = compute_node.src_a.tile_shape.face_r_dim
    num_faces_a = compute_node.src_a.tile_shape.total_num_faces()
    tile_size_a = compute_node.src_a.tile_size

    if compute_node.src_b is not None:
        face_r_dim_b = compute_node.src_b.tile_shape.face_r_dim
        num_faces_b = compute_node.src_b.tile_shape.total_num_faces()
        tile_size_b = compute_node.src_b.tile_size
    else:
        face_r_dim_b = face_r_dim_a
        num_faces_b = num_faces_a
        tile_size_b = tile_size_a

    return (
        f"_llk_unpack_hw_configure_<{dest_acc}>(\n"
        f"    {unpack_A_src.cpp_underlying_value}, {unpack_B_src.cpp_underlying_value},\n"
        f"    {unpack_A_dst.cpp_underlying_value}, {unpack_B_dst.cpp_underlying_value},\n"
        f"    {face_r_dim_a}, {face_r_dim_b}, {num_faces_a}, {num_faces_b},\n"
        f"    {tile_size_a}, {tile_size_b}\n"
        f");\n"
    )


def configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    old_A_src: DataFormat,
    new_A_src: DataFormat,
    new_A_dst: DataFormat,
    old_B_src: DataFormat,
    new_B_src: DataFormat,
    new_B_dst: DataFormat,
    srca_changed: bool,
    srcb_changed: bool,
    srca_tile_changed: bool,
    srcb_tile_changed: bool,
) -> str:
    code = ""

    if srca_changed:
        new_face_r_dim_a = compute_node.src_a.tile_shape.face_r_dim
        new_num_faces_a = compute_node.src_a.tile_shape.total_num_faces()

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
            f"    {new_A_src.cpp_underlying_value}, {new_A_dst.cpp_underlying_value}, {compute_node.src_a.tile_size}"
        )
        if srca_tile_changed:
            code += f", {new_face_r_dim_a}, {new_num_faces_a}"
        code += "\n);\n"

    if srcb_changed:
        if compute_node.reuse_dest is EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            srcb_tile_size = compute_node.src_a.tile_size
        elif compute_node.src_b is not None:
            srcb_tile_size = compute_node.src_b.tile_size
        else:
            srcb_tile_size = None

        if srcb_tile_size is not None:
            if compute_node.src_b is not None:
                new_face_r_dim_b = compute_node.src_b.tile_shape.face_r_dim
                new_num_faces_b = compute_node.src_b.tile_shape.total_num_faces()
            else:
                new_face_r_dim_b = compute_node.src_a.tile_shape.face_r_dim
                new_num_faces_b = compute_node.src_a.tile_shape.total_num_faces()

            to_from_int8 = (
                "true"
                if old_B_src.needs_int8_math_config()
                or new_B_src.needs_int8_math_config()
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


def sync_with_packer(needs_pack_sync: bool) -> str:
    if needs_pack_sync:
        return (
            "t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);\n"
            "t6_semaphore_get<>(semaphore::PACK_DONE);\n"
        )
    return ""
