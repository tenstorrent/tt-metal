# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.operand import Operand
from fuser.quasar import dest_dvalid
from helpers.format_config import DataFormat
from helpers.golden_generators import PackGolden

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation
    from fuser.pack_node import PackNode


def hw_configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
    pack_mode: str = "PackMode::Default",
) -> str:
    desc = output.cpp_desc_name
    return (
        f"{desc}.reg_data_format = static_cast<std::uint8_t>({pack_src.cpp_underlying_value});\n"
        f"_llk_pack_hw_configure_<p_pacr::PACK0, {dest_acc}>({desc}, ReluConfig::from_packed(0));\n"
    )


def configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    desc = output.cpp_desc_name
    return (
        f"{desc}.reg_data_format = static_cast<std::uint8_t>({pack_src.cpp_underlying_value});\n"
        f"_llk_pack_hw_configure_<p_pacr::PACK0, {dest_acc}>({desc}, ReluConfig::from_packed(0));\n"
    )


def relu_config(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    dest_acc = config.dest_acc.cpp_enum_value
    pack_src_format = config.sentinel._pack_src

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
        return dest_dvalid.enable(config, operation, dest_dvalid.PACK_THREAD)
    if operation.stage_id != 1:
        return ""
    return f"_llk_pack_dest_init_<p_pacr::PACK0, {operation.dest_sync.cpp_enum_value}>();\n"


def packer_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return "_llk_packer_wait_for_math_done_();\n"


def packer_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.quasar_use_dvalid:
        return dest_dvalid.signal(
            config, operation, dest_dvalid.PACK_THREAD, dest_dvalid.PACK
        )
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, {dest_sync}, {dest_acc}>();\n"


def dvalid_signal_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    return dest_dvalid.signal(
        config, operation, dest_dvalid.PACK_THREAD, dest_dvalid.SFPU
    )


def dvalid_disable(config: "GlobalConfig", operation: "L1Operation") -> str:
    return dest_dvalid.disable(config, operation, dest_dvalid.PACK_THREAD)


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
