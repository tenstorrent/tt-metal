# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData, InvocationGranularity
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation


class UnpackerTilizeA(Unpacker):
    granularity = InvocationGranularity.TILE

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
        valid_cnt = 1
        return f"_perf_unpack_loop_set_valid<true, true>({valid_cnt});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        valid_cnt = 1
        return f"_perf_math_loop_clear_valid<true, true>({valid_cnt});\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.tilize_golden(tensor_a, config, operation, compute_unit),
            None,
        )

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
        block_ct_dim = compute_unit.src_a.tile_count_x
        buffer_a = compute_unit.src_a.cpp_name

        return (
            f"{{\n"
            f"    std::uint32_t row = ({block.tile_id_global}) / {block_ct_dim};\n"
            f"    std::uint32_t col = ({block.tile_id_global}) % {block_ct_dim};\n"
            f"    _llk_unpack_tilize_(L1_ADDRESS({buffer_a}[row * {block_ct_dim}]), col, {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format});\n"
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
