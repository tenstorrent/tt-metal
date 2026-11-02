# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.pack.untilize import untilize_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.pack_node import PackNode
from helpers.format_config import DataFormat

from .common import untilize_l1_address
from .packer import Packer


class PackUntilize(Packer):
    granularity = InvocationGranularity.ROW

    golden_fn = staticmethod(untilize_golden)

    per_block_init = True
    pack_mode = "PackMode::Untilize"
    requires_dest_remap = True

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
            "llk_pack_untilize.h",
        ]

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        block_ct_dim = block.block_cols
        full_ct_dim = pack_node.output.tile_count_x
        face_r_dim = pack_node.output.tile_shape.face_r_dim
        num_faces = pack_node.output.tile_shape.total_num_faces()

        # Non-8-bit Blackhole tilize leaves adjacent left/right face rows in
        # Dest; ordinary untilize instead expects face-major rows.
        src_is_tilize = False
        if operation.bh_tilize.value:
            byte_formats = {DataFormat.Fp8_e4m3, DataFormat.Int8, DataFormat.UInt8}
            layouts = {
                isinstance(source, FpuNode)
                and source.src_a is not None
                and source.src_a.data_format not in byte_formats
                for source in config.sentinel.dest_sources_for(pack_node)
            }
            if len(layouts) > 1:
                raise ValueError(
                    "PackUntilize requires a single Blackhole tilize source layout"
                )
            src_is_tilize = next(iter(layouts), False)
        return (
            f"_llk_pack_untilize_init_<{block_ct_dim}, {full_ct_dim}, "
            f"false, TILE_C_DIM, false, {str(src_is_tilize).lower()}>(\n"
            f"    {config.sentinel.pack_src_format}, {config.sentinel.pack_dst_format}, {face_r_dim}, {num_faces}\n"
            f");\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        block_ct_dim = block.block_cols
        full_ct_dim = pack_node.output.tile_count_x
        num_faces = pack_node.output.tile_shape.total_num_faces()

        return (
            f"_llk_pack_untilize_<{block_ct_dim}, {full_ct_dim}>(\n"
            f"    {untilize_l1_address(pack_node.output, block)},\n"
            f"    {num_faces}, {block.tile_id_dest}\n"
            f");\n"
        )

    def uninit(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return f"_llk_pack_untilize_uninit_({config.sentinel.pack_src_format});\n"
