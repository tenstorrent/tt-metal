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
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import DataFormat, ReduceDimension, ReducePool
from helpers.tilize_untilize import tilize_block, untilize_block


class ReduceFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        dimensions = operation.max_output_dimensions
        tile_elements = operation.tile_shape.total_tile_size()
        tile_cnt = (dimensions[0] * dimensions[1]) // tile_elements
        num_faces = operation.tile_shape.total_num_faces()

        reduce_dim = self.reduce_dim
        pool_type = self.reduce_pool

        tile_dims = (
            operation.tile_shape.total_row_dim(),
            operation.tile_shape.total_col_dim(),
        )
        src_a_reduced_tensor = tilize_block(
            tensor_a,
            dimensions,
            output_format,
            num_faces,
            tile_dimensions=tile_dims,
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        src_a_reduced_tensor = generate_golden(
            src_a_reduced_tensor,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
            tile_shape=operation.tile_shape,
        ).flatten()

        if compute_unit.reduce_to_tile:
            block_tiles = operation.block_tiles_x * operation.block_tiles_y
            tiles = src_a_reduced_tensor.view(-1, tile_elements)
            for b in range(0, tile_cnt, block_tiles):
                block = tiles[b : b + block_tiles]
                if pool_type == ReducePool.Max:
                    tiles[b] = block.max(dim=0).values
                else:
                    tiles[b] = block.sum(dim=0)
                tiles[b + 1 : b + block_tiles] = 0

        dest_golden_tensor = tilize_block(
            tensor_dst,
            dimensions,
            output_format,
            num_faces,
            tile_dimensions=tile_dims,
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        dest_golden_tensor = generate_golden(
            dest_golden_tensor,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
            tile_shape=operation.tile_shape,
        ).flatten()

        total_elements = tile_cnt * tile_elements
        golden_tensor = torch.zeros(total_elements)

        reduce_dim_size = (
            operation.tile_shape.total_col_dim()
            if self.reduce_dim == ReduceDimension.Row
            else operation.tile_shape.total_row_dim()
        )

        for i in range(total_elements):
            if self.reduce_pool == ReducePool.Max:
                golden_tensor[i] = max(src_a_reduced_tensor[i], dest_golden_tensor[i])
            if self.reduce_pool == ReducePool.Sum:
                golden_tensor[i] = src_a_reduced_tensor[i] + dest_golden_tensor[i]
            if self.reduce_pool == ReducePool.Average:
                golden_tensor[i] = (
                    src_a_reduced_tensor[i] * reduce_dim_size + dest_golden_tensor[i]
                ) / reduce_dim_size

        golden_tensor = untilize_block(
            golden_tensor,
            output_format,
            dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: FusedOperation,
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
            f"_llk_math_reduce_init_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}>(\n"
            f"{compute_unit.src_a.tile_shape.cpp_value});\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
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
            f"{block.tile_id_block}, {compute_unit.src_a.tile_shape.cpp_value}\n"
            f");\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_math_reduce_uninit_();\n"
