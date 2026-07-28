# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fused_operand import Operand
from fuser.wormhole.packer.common import (  # noqa: F401
    configure_pack,
    l1_accumulation_config,
    pack_dest_init,
    packer_dest_section_done,
    packer_sync_with_unpacker,
    packer_wait_for_math,
    relu_config,
)
from helpers.format_config import DataFormat


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
