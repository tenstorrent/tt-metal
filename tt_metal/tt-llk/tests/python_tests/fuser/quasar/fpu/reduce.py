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


def _is_int_fpu_enabled(
    reduce_dim: ReduceDimension, config: GlobalConfig, compute_unit: FpuNode
) -> str:
    int_fpu_formats = {DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32}
    return (
        "true"
        if (
            config.dest_acc.value
            and reduce_dim == ReduceDimension.Row
            and any(
                operand is not None and operand.data_format in int_fpu_formats
                for operand in (compute_unit.src_a, compute_unit.src_b)
            )
        )
        else "false"
    )


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
        pool_type_cpp = self.reduce_pool.cpp_enum_value
        reduce_dim_cpp = self.reduce_dim.cpp_enum_value
        is_int_fpu_en = _is_int_fpu_enabled(self.reduce_dim, config, compute_unit)
        return (
            f"// Operation {stage}: Reduce {reduce_dim_cpp} FPU\n"
            f"_llk_math_reduce_init_<{pool_type_cpp}, {reduce_dim_cpp}, {math_fidelity}, {is_int_fpu_en}>"
            f"({compute_unit.src_a.tile_shape.cpp_value});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        pool_type_cpp = self.reduce_pool.cpp_enum_value
        reduce_dim_cpp = self.reduce_dim.cpp_enum_value
        is_int_fpu_en = _is_int_fpu_enabled(self.reduce_dim, config, compute_unit)
        return (
            f"_llk_math_reduce_<{pool_type_cpp}, {reduce_dim_cpp}, {is_int_fpu_en}>"
            f"({block.tile_id_block}, {compute_unit.src_a.tile_shape.cpp_value});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
