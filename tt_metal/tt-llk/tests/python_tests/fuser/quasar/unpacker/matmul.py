# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_loop import FusedLoop, LoopBlock
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig


class MatmulUnpacker(Unpacker):
    loop: FusedLoop = LoopBlock()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_matmul.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id_a = compute_unit.src_a.buf_desc_id
        buf_desc_id_b = compute_unit.src_b.buf_desc_id
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols

        return (
            f"_llk_unpack_matmul_init_<false>"
            f"({buf_desc_id_a}, {buf_desc_id_b}, {ct_dim}, {rt_dim}, {kt_dim});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        full_ct_dim = (
            compute_unit.src_b.dimensions[1]
            // compute_unit.src_b.tile_shape.total_col_dim()
        )
        output_ct_dim = compute_unit.src_a.tile_count_x

        return (
            f"{{\n"
            f"    std::uint32_t row = ({block.tile_id_global}) / {output_ct_dim};\n"
            f"    std::uint32_t col = ({block.tile_id_global}) % {output_ct_dim};\n"
            f"    for (std::uint32_t kt = 0; kt < {kt_dim}; ++kt) {{\n"
            f"        std::uint32_t srca_tile_idx = row * {kt_dim} + kt;\n"
            f"        std::uint32_t srcb_tile_idx = kt * {full_ct_dim} + col;\n"
            f"        _llk_unpack_matmul_({ct_dim}, {rt_dim}, {kt_dim}, srca_tile_idx, srcb_tile_idx);\n"
            f"    }}\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
