# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fpu_node import FpuNode
from fuser.fused_operand import Operand
from helpers.format_config import DataFormat


def is_datacopy_node(compute_node: FpuNode) -> bool:
    from fuser.quasar.fpu.datacopy import DatacopyFpu

    return isinstance(compute_node.fpu, DatacopyFpu)


def is_unary_unpacker(compute_node: FpuNode) -> bool:
    from fuser.quasar.unpacker.unpack_a import UnpackerA

    return isinstance(compute_node.unpacker, UnpackerA)


def _emit_buf_desc(
    var_prefix: str,
    operand: Operand,
    src_fmt: DataFormat,
    dst_fmt: DataFormat,
) -> str:
    code = (
        f"tdma_descriptor_t td_{var_prefix} = ckernel::trisc::construct_tdma_desc("
        f"{operand.tile_shape.cpp_value}, "
        f"{operand.cpp_name}[0] / 16, "
        f"{src_fmt.cpp_underlying_value}, "
        f"{operand.buf_desc_id}, "
        f"{dst_fmt.cpp_underlying_value});\n"
        f"_configure_buf_desc_table_(td_{var_prefix}.buf_desc_id, td_{var_prefix}.buf_desc);\n"
    )
    return code


def _emit_configure(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_src: DataFormat,
    unpack_A_dst: DataFormat,
    unpack_B_src: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    is_unary = is_unary_unpacker(compute_node)
    is_datacopy_dest_acc = is_datacopy_node(compute_node) and dest_acc == "true"

    code = _emit_buf_desc("val_A", compute_node.src_a, unpack_A_src, unpack_A_dst)

    if is_unary and is_datacopy_dest_acc:
        code += "_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_A);\n"
    elif is_unary:
        code += "_llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val_A);\n"
    else:
        code += _emit_buf_desc("val_B", compute_node.src_b, unpack_B_src, unpack_B_dst)
        code += "_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);\n"

    return code


def hw_configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_src: DataFormat,
    unpack_A_dst: DataFormat,
    unpack_B_src: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    return _emit_configure(
        compute_node, dest_acc, unpack_A_src, unpack_A_dst, unpack_B_src, unpack_B_dst
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
    code = "{\n"
    code += _emit_configure(
        compute_node, dest_acc, new_A_src, new_A_dst, new_B_src, new_B_dst
    )
    code += "}\n"
    return code


def dvalid_init() -> str:
    return "set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"


def sync_with_packer(stage_id: int) -> str:
    return ""
