# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.fpu_node import FpuNode
from helpers.format_config import DataFormat

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def hw_configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_src: DataFormat,
    unpack_A_dst: DataFormat,
    unpack_B_src: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    return (
        f"_llk_unpack_hw_configure_<p_unpacr::UNP_A>(static_cast<DataFormat>({unpack_A_dst.cpp_underlying_value}));\n"
        f"_llk_unpack_hw_configure_<p_unpacr::UNP_B>(static_cast<DataFormat>({unpack_B_dst.cpp_underlying_value}));\n"
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
    if srca_changed and srcb_changed:
        return f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(static_cast<DataFormat>({new_A_dst.cpp_underlying_value}), static_cast<DataFormat>({new_B_dst.cpp_underlying_value}));\n"
    if srca_changed:
        return f"_llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>({new_A_dst.cpp_underlying_value}));\n"
    if srcb_changed:
        return f"_llk_unpack_configure_unary_<p_unpacr::UNP_B>(static_cast<DataFormat>({new_B_dst.cpp_underlying_value}));\n"
    return ""


def dvalid_init(config: "GlobalConfig" = None, operation: "L1Operation" = None) -> str:
    from helpers.llk_params import PerfRunType

    if config.quasar_use_dvalid:
        if config.perf_run_type in (None, PerfRunType.L1_TO_L1):
            return "set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"
        return "set_up_zero_dest_dvalid_handshake_for_unpack();\n"
    return ""


def sync_with_packer(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.needs_pack_sync:
        return (
            "_llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::PACK_UNPACK_L1);\n"
            "_llk_sync_get_<>(semaphore::PACK_UNPACK_L1);\n"
        )
    return ""


def unpack_to_dest_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not config.uses_unpack_to_dest_sync or operation.stage_id != 1:
        return ""
    return (
        "set_up_zero_dest_dvalid_handshake_for_unpack();\n"
        "_llk_unpack_dest_init_();\n"
    )


def unpack_to_dest_wait_for_dest_available(config: "GlobalConfig") -> str:
    if not config.uses_unpack_to_dest_sync:
        return ""
    return "_llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();\n"


def unpack_to_dest_section_done(
    config: "GlobalConfig", operation: "L1Operation"
) -> str:
    if not config.uses_unpack_to_dest_sync:
        return ""
    return "_llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();\n"
