# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.block_data import BlockData
from fuser.operand import Operand
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
    return (
        f"_llk_pack_hw_configure_<{dest_acc}, PackMode::Default>(\n"
        f"    {pack_src.cpp_underlying_value}, {pack_dst.cpp_underlying_value}, {output.tile_size}, {output.tile_shape.face_r_dim}, {output.tile_shape.total_num_faces()}\n"
        f");\n"
    )


def configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    return (
        f"_llk_pack_reconfig_data_format_<{dest_acc}>(\n"
        f"    {pack_src.cpp_underlying_value}, {pack_dst.cpp_underlying_value}, {output.tile_size}\n"
        f");\n"
    )


def relu_config(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    pack_src_format = config.sentinel._pack_src

    relu_config_val = PackGolden.generate_relu_config(
        node.pack_relu, node.relu_threshold, pack_src_format
    )
    return f"_llk_pack_relu_config_(ReluConfig::from_packed({relu_config_val}));\n"


def l1_accumulation_config(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    l1_acc = node.pack_l1_accumulation.cpp_enum_value
    return f"_llk_pack_reconfig_l1_acc_({l1_acc});\n"


def untilize_l1_address(output: Operand, block: BlockData) -> str:
    tile_size_16B = output.tile_size
    row_stride = output.tile_count_x * tile_size_16B
    col_stride = tile_size_16B // output.tile_shape.total_row_dim()

    return (
        f"L1_ADDRESS({output.cpp_name}[0])"
        f" + {row_stride} * ({block.block_y} + tile_y)"
        f" + {col_stride} * {block.block_x}"
    )


def pack_dest_init(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    if operation.stage_id != 1:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    pack_mode = node.packer.pack_mode
    return f"_llk_pack_dest_init_<{dest_sync}, {dest_acc}, {pack_mode}>();\n"


def packer_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    return "_llk_packer_wait_for_math_done_();\n"


def packer_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_pack_dest_section_done_<{dest_sync}, {dest_acc}>();\n"


def packer_sync_with_unpacker(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.has_pack_consumer:
        return "t6_semaphore_post<>(semaphore::PACK_DONE);\n\n"
    return ""


def pack_reduce_mask_config(operation: "L1Operation") -> str:
    if operation.reduce_dim is None:
        return ""
    reduce_dim = operation.reduce_dim.cpp_enum_value
    return f"_llk_pack_reduce_mask_config_<{reduce_dim}>();\n"


def pack_reduce_mask_clear(operation) -> str:
    if operation.reduce_dim is None:
        return ""
    return "_llk_pack_reduce_mask_clear_();\n"
