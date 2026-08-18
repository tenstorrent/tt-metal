# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.operand import Operand
from fuser.wormhole.packer.common import (  # noqa: F401
    configure_pack,
    l1_accumulation_config,
    pack_reduce_mask_clear,
    pack_reduce_mask_config,
    packer_dest_section_done,
    packer_sync_with_unpacker,
    packer_wait_for_math,
    relu_config,
    untilize_l1_address,
)
from helpers.format_config import DataFormat

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation
    from fuser.pack_node import PackNode


def pack_dest_init(
    config: "GlobalConfig", operation: "L1Operation", node: "PackNode"
) -> str:
    if operation.stage_id != 1:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_pack_dest_init_<{dest_sync}, {dest_acc}>();\n"


def hw_configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
    pack_mode: str = "PackMode::Default",
) -> str:
    return (
        f"_llk_pack_hw_configure_<{dest_acc}, {pack_mode}>(\n"
        f"    {pack_src.cpp_underlying_value}, {pack_dst.cpp_underlying_value}, {output.tile_size}, {output.tile_shape.face_r_dim}, {output.tile_shape.total_col_dim()}, {output.tile_shape.total_num_faces()}\n"
        f");\n"
    )
