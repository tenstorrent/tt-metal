# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.reduce_tilize_a import reduce_tilize_a_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from helpers.llk_params import ReducePool


class UnpackReduceTilize(Unpacker):
    granularity = InvocationGranularity.TILE

    golden_fn = staticmethod(reduce_tilize_a_golden)

    def __init__(self, reduce_pool: ReducePool):
        self.reduce_pool = reduce_pool

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def _template_args(self) -> str:
        neginf_src_a = str(self.reduce_pool == ReducePool.Max).lower()
        zero_src_a_reduce = str(self.reduce_pool != ReducePool.Max).lower()
        return f"<{neginf_src_a}, true /*reload_srcB*/, false /*zero_srcA*/, {zero_src_a_reduce}>"

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_perf_unpack_loop_set_valid<true, true>(1);\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_perf_math_loop_clear_valid<true, true>(1);\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        block_ct_dim = compute_unit.src_a.tile_count_x
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        unpb_face_r_dim = compute_unit.src_b.tile_shape.face_r_dim

        return (
            f"_llk_unpack_tilizeA_B_init_{self._template_args()}("
            f"{config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}, "
            f"{block_ct_dim}, {num_faces}, {unpb_face_r_dim});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        block_ct_dim = compute_unit.src_a.tile_count_x
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name

        return (
            f"{{\n"
            f"std::uint32_t row = ({block.tile_id_src_a}) / {block_ct_dim};\n"
            f"std::uint32_t col = ({block.tile_id_src_a}) % {block_ct_dim};\n"
            f"_llk_unpack_tilizeA_B_{self._template_args()}("
            f"{config.sentinel.unpack_a_src_format}, {face_r_dim}, "
            f"L1_ADDRESS({buffer_a}[row * {block_ct_dim}]), "
            f"L1_ADDRESS({buffer_b}[{block.tile_id_src_b}]), "
            f"col, {block_ct_dim}, {num_faces});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return (
            f"_llk_unpack_tilizeA_B_uninit_({config.sentinel.unpack_a_dst_format});\n"
        )
