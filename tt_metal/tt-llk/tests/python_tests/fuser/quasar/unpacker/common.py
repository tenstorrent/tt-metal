# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fpu_node import FpuNode
from helpers.format_config import DataFormat
from helpers.llk_params import EltwiseBinaryReuseDestType


def is_datacopy_node(compute_node: FpuNode) -> bool:
    from fuser.quasar.fpu.datacopy import DatacopyFpu

    return isinstance(compute_node.fpu, DatacopyFpu)


def is_unary_unpacker(compute_node: FpuNode) -> bool:
    from fuser.quasar.unpacker.unpack_a import UnpackerA

    return isinstance(compute_node.unpacker, UnpackerA)


def _emit_configure(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_dst: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    is_unary = is_unary_unpacker(compute_node)
    is_datacopy_dest_acc = is_datacopy_node(compute_node) and dest_acc == "true"
    has_reuse_dest = compute_node.reuse_dest != EltwiseBinaryReuseDestType.NONE

    desc_a = compute_node.src_a.cpp_desc_name
    code = f"{desc_a}.reg_data_format = static_cast<std::uint8_t>({unpack_A_dst.cpp_underlying_value});\n"

    if is_unary and (is_datacopy_dest_acc or has_reuse_dest):
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


def dvalid_init(sfpu_on_dest: bool = False) -> str:
    chain = "DestChain::FPU_SFPU_PACK" if sfpu_on_dest else "DestChain::FPU_PACK"
    return f"setup_dest_dvalid<{chain}>();\n"


def sync_with_packer(needs_pack_sync: bool) -> str:
    if needs_pack_sync:
        return (
            "TT_SEMWAIT(p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO, 0, "
            "semaphore::t6_sem(semaphore::PACK_DONE));\n"
            "t6_semaphore_get<>(semaphore::PACK_DONE);\n"
        )
    return ""
