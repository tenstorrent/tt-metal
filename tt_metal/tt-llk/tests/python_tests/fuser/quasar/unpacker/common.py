# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.fpu_node import FpuNode
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
        return f"_llk_unpack_configure_unary_<p_unpacr::UNP_B>(static_cast<DataFormat>({unpack_B_dst.cpp_underlying_value}));\n"

    is_unary = is_unary_unpacker(compute_node)
    has_reuse_dest = compute_node.reuse_dest != EltwiseBinaryReuseDestType.NONE
    unpack_to_dest = compute_node.unpack_to_dest.value

    code = ""
    if unpack_to_dest:
        code += f"_llk_math_upk_to_dest_hw_configure_<false, {dest_acc}, false>();\n"

    if is_unary and ((dest_acc == "true" and not unpack_to_dest) or has_reuse_dest):
        code += f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(static_cast<DataFormat>({unpack_A_dst.cpp_underlying_value}), static_cast<DataFormat>({unpack_A_dst.cpp_underlying_value}));\n"
    elif is_unary:
        code += f"_llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>({unpack_A_dst.cpp_underlying_value}));\n"
    else:
        code += f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(static_cast<DataFormat>({unpack_A_dst.cpp_underlying_value}), static_cast<DataFormat>({unpack_B_dst.cpp_underlying_value}));\n"

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


def _upk_to_dest_sem_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    from fuser.fpu_node import FpuNode
    from fuser.quasar.unpacker.unpack_a import _uses_upk_to_dest_semaphores

    if not _uses_upk_to_dest_semaphores(config):
        return ""
    if not any(
        isinstance(node, FpuNode) and node.unpack_to_dest.value
        for node in operation.math.math_nodes
    ):
        return ""
    return "_llk_sync_init_(semaphore::UNPACK_MATH, 1, 0);\n"


def dvalid_init(config: "GlobalConfig" = None, operation: "L1Operation" = None) -> str:
    from helpers.llk_params import PerfRunType

    if config.quasar_use_dvalid:
        if config.perf_run_type in (None, PerfRunType.L1_TO_L1):
            return "set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"
        return "set_up_zero_dest_dvalid_handshake_for_unpack();\n"
    return _upk_to_dest_sem_init(config, operation)


def sync_with_packer(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.needs_pack_sync:
        return (
            "_llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::PACK_UNPACK);\n"
            "_llk_sync_get_<>(semaphore::PACK_UNPACK);\n"
        )
    return ""
