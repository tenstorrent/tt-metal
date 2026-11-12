# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.golden_generators import PackGolden
from helpers.llk_params import DestSync, PerfRunType

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation
    from fuser.pack_node import PackNode


def hw_configure_pack(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    pack_src = node.pack_src_format
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_pack_hw_configure_<p_pacr::PACK0, {dest_acc}>(static_cast<DataFormat>({pack_src.cpp_underlying_value}), ReluConfig::from_packed(0));\n"


def configure_pack(
    config: "GlobalConfig", node: "PackNode", previous: "PackNode"
) -> str:
    if not node.pack_changed(previous):
        return ""
    pack_src, pack_dst = node.pack_src_format, node.pack_dst_format
    return f"_llk_pack_reconfig_data_format_<p_pacr::PACK0>({pack_src.cpp_underlying_value}, {pack_dst.cpp_underlying_value});\n"


def relu_config(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    dest_acc = config.dest_acc.cpp_enum_value
    pack_src_format = node.pack_src_format

    relu_config_val = PackGolden.generate_relu_config(
        node.pack_relu, node.relu_threshold, pack_src_format
    )
    return f"_llk_pack_relu_config_<p_pacr::PACK0, {dest_acc}>(ReluConfig::from_packed({relu_config_val}));\n"


def l1_accumulation_config(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    l1_acc = node.pack_l1_accumulation.cpp_enum_value
    return f"_llk_pack_set_l1_acc_<p_pacr::PACK0>({l1_acc});\n"


def pack_dest_init(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    if config.quasar_use_dvalid:
        if config.perf_run_type in (None, PerfRunType.L1_TO_L1):
            return "set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"
        return "set_up_zero_dest_dvalid_handshake_for_pack();\n"
    if operation.stage_id != 1:
        return ""
    code = f"_llk_pack_dest_init_<p_pacr::PACK0, {operation.dest_sync.cpp_enum_value}>();\n"
    if config.uses_unpack_to_dest_sync:
        code = "set_up_zero_dest_dvalid_handshake_for_pack();\n" + code
    return code


def packer_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return "_llk_packer_wait_for_math_done_();\n"


def packer_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.uses_unpack_to_dest_pack_sync:
        code = (
            "_llk_sync_get_<p_stall::PACK0>(semaphore::UNPACK_PACK);\n"
            "_llk_sync_get_<p_stall::PACK0>(semaphore::MATH_PACK);\n"
        )
        if operation.dest_sync == DestSync.Half:
            code += (
                f"_update_dest_register_offset_<{config.dest_acc.cpp_enum_value}>();\n"
                "_llk_stall_cfg_on_<p_stall::PACK0>();\n"
                f"_set_packer_dest_registers_<p_pacr::PACK0, {operation.dest_sync.cpp_enum_value}>();\n"
            )
        return code
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    if config.quasar_use_dvalid:
        return f"_llk_pack_dest_dvalid_section_done_<{dest_sync}, {dest_acc}>();\n"
    return f"_llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, {dest_sync}, {dest_acc}>();\n"


def packer_sync_with_unpacker(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.has_pack_consumer:
        return "_llk_sync_post_<p_stall::PACK>(semaphore::PACK_UNPACK);\n"
    return ""


def pack_reduce_mask_config(operation) -> str:
    if operation.reduce_dim is None:
        return ""
    reduce_dim = operation.reduce_dim.cpp_enum_value
    tensor_shape = operation.tile_shape.cpp_value
    return f"_llk_pack_reduce_mask_config_<{reduce_dim}>({tensor_shape});\n"


def pack_reduce_mask_clear(operation) -> str:
    if operation.reduce_dim is None:
        return ""
    return "_llk_pack_reduce_mask_clear_();\n"
