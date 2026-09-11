# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.state import OutputLayout
from fuser.golden.unpack.tilize_a import tilize_a_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation


class UnpackerTilizeA(Unpacker):
    granularity = InvocationGranularity.TILE

    output_layout = OutputLayout.TILED

    golden_fn = staticmethod(tilize_a_golden)

    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        valid_cnt = 4
        return f"_perf_unpack_loop_set_valid<true, true>({valid_cnt});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        valid_cnt = 4
        return f"_perf_math_loop_clear_valid<true, true>({valid_cnt});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        block_ct_dim = compute_unit.src_a.tile_count_x

        return f"_llk_unpack_tilize_init_({config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}, {block_ct_dim}, {face_r_dim}, false);\n"

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        block_ct_dim = compute_unit.src_a.tile_count_x
        buffer_a = compute_unit.src_a.cpp_name

        return (
            f"{{\n"
            f"std::uint32_t row = ({block.tile_id_src_a}) / {block_ct_dim};\n"
            f"std::uint32_t col = ({block.tile_id_src_a}) % {block_ct_dim};\n"
            f"_llk_unpack_tilize_(L1_ADDRESS({buffer_a}[row * {block_ct_dim}]), col, {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}, {block_ct_dim}, {face_r_dim}, {num_faces}, false);\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value

        return f"_llk_unpack_tilize_uninit_({config.sentinel.unpack_a_dst_format}, {tensor_shape});\n"
