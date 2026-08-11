# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.tile_loop import LoopTileByTile, TileLoop
from helpers.llk_params import DataFormat, ReduceDimension, ReducePool


class ReduceFpu(Fpu):
    loop: TileLoop = LoopTileByTile()

    def __init__(self, reduce_dim: ReduceDimension, reduce_pool: ReducePool):
        self.reduce_dim = reduce_dim
        self.reduce_pool = reduce_pool

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_reduce.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reduce_golden(
            tensor_a, tensor_b, tensor_dst, config, operation, compute_unit
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        pool_type_cpp = self.reduce_pool.cpp_enum_value
        reduce_dim_cpp = self.reduce_dim.cpp_enum_value

        return (
            f"// Operation {stage}: Reduce {reduce_dim_cpp} FPU\n"
            f"_llk_math_reduce_init_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}>({compute_unit.src_a.tile_shape.cpp_value});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        pool_type_cpp = self.reduce_pool.cpp_enum_value
        reduce_dim_cpp = self.reduce_dim.cpp_enum_value

        _int_fpu_formats = {DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32}
        is_int_fpu_en = (
            "true"
            if (
                (
                    compute_unit.src_a is not None
                    and compute_unit.src_a.data_format in _int_fpu_formats
                )
                or (
                    compute_unit.src_b is not None
                    and compute_unit.src_b.data_format in _int_fpu_formats
                )
            )
            else "false"
        )

        return (
            f"_llk_math_reduce_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, {is_int_fpu_en}>(\n"
            f"    {block.tile_id_block}, {compute_unit.src_a.tile_shape.cpp_value}\n"
            f");\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_reduce_uninit_();\n"
