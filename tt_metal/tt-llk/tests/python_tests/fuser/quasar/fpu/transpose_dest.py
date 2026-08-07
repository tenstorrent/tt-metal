# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import TransposeGolden, get_golden_generator


class TransposeDestFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_transpose_dest.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        tile_cnt = compute_unit.src_a.tile_count

        t_matrix = get_golden_generator(TransposeGolden)
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            tensor_dst,
            output_format,
            num_tiles=tile_cnt,
            tilize=True,
            input_dimensions=compute_unit.src_a.dimensions,
        )
        golden_tensor = t_matrix.transpose_within_faces_multi_tile(
            golden_tensor,
            output_format,
            num_tiles=tile_cnt,
            untilize=True,
            input_dimensions=compute_unit.src_a.dimensions,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        en_32bit_dest = config.dest_acc.cpp_enum_value
        math_format = config.sentinel._math_format.cpp_enum_value
        return (
            f"_configure_mov_ops_explicit_alu_data_format_state_<{en_32bit_dest}>({math_format}, {math_format});"
            f"_llk_math_transpose_dest_init_<true, {en_32bit_dest}>();\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_math_transpose_dest_({block.tile_id_block});\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
