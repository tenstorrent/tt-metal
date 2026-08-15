# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.fpu_node import FpuNode
from fuser.quasar.fpu.common import math_writes_dest, unpack_writes_dest
from helpers.format_config import DataFormat
from helpers.llk_params import EltwiseBinaryReuseDestType

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def is_unary_unpacker(compute_node: FpuNode) -> bool:
    from fuser.quasar.unpacker.tilize_a import UnpackerTilizeA
    from fuser.quasar.unpacker.unpack_a import UnpackerA

    return isinstance(compute_node.unpacker, (UnpackerA, UnpackerTilizeA))


def _is_unary_broadcast_unpacker(compute_node: FpuNode) -> bool:
    from fuser.quasar.unpacker.unary_broadcast import UnaryBroadcastUnpacker

    return isinstance(compute_node.unpacker, UnaryBroadcastUnpacker)


def _emit_configure(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_dst: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    if _is_unary_broadcast_unpacker(compute_node):
        desc_b = compute_node.src_b.cpp_desc_name
        code = f"{desc_b}.reg_data_format = static_cast<std::uint8_t>({unpack_B_dst.cpp_underlying_value});\n"
        code += f"_llk_unpack_configure_unary_<p_unpacr::UNP_B>({desc_b});\n"
        return code

    is_unary = is_unary_unpacker(compute_node)
    has_reuse_dest = compute_node.reuse_dest != EltwiseBinaryReuseDestType.NONE
    unpack_to_dest = compute_node.unpack_to_dest.value

    desc_a = compute_node.src_a.cpp_desc_name
    code = f"{desc_a}.reg_data_format = static_cast<std::uint8_t>({unpack_A_dst.cpp_underlying_value});\n"
    if unpack_to_dest:
        code += f"_llk_math_upk_to_dest_hw_configure_<false, {dest_acc}, false>();\n"

    if is_unary and ((dest_acc == "true" and not unpack_to_dest) or has_reuse_dest):
        code += f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>({desc_a}, {desc_a});\n"
    elif is_unary:
        code += f"_llk_unpack_configure_unary_<p_unpacr::UNP_A>({desc_a});\n"
    else:
        desc_b = compute_node.src_b.cpp_desc_name
        code += f"{desc_b}.reg_data_format = static_cast<std::uint8_t>({unpack_B_dst.cpp_underlying_value});\n"
        code += f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>({desc_a}, {desc_b});\n"

    return code


def hw_configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_src: DataFormat,
    unpack_A_dst: DataFormat,
    unpack_B_src: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    return _emit_configure(compute_node, dest_acc, unpack_A_dst, unpack_B_dst)


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
    return _emit_configure(compute_node, dest_acc, new_A_dst, new_B_dst)


def dvalid_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not config.quasar_use_dvalid:
        return ""
    if not unpack_writes_dest(operation):
        return "_llk_dest_dvalid_disable_<dest_dvalid::client::UNPACK>();\n"
    successor = "FPU" if math_writes_dest(operation) else "PACK"
    return (
        "_llk_dest_dvalid_configure_<dest_dvalid::client::UNPACK, "
        f"dest_dvalid::client::{successor}, true>();\n"
    )


def dvalid_signal(
    config: "GlobalConfig", operation: "L1Operation", compute_node: FpuNode
) -> str:
    if not config.quasar_use_dvalid or not compute_node.unpack_to_dest.value:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return (
        "_llk_dest_dvalid_signal_<dest_dvalid::client::UNPACK, "
        f"{dest_sync}, {dest_acc}>();\n"
    )


def sync_with_packer(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.needs_pack_sync:
        return (
            "_llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::PACK_UNPACK);\n"
            "_llk_sync_get_<>(semaphore::PACK_UNPACK);\n"
        )
    return ""
