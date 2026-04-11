# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    ReducePool,
)
from helpers.tilize_untilize import tilize_block, untilize_block


class ReduceFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

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
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format
        dimensions = operation.max_output_dimensions
        tile_cnt = (dimensions[0] * dimensions[1]) // 1024
        num_faces = operation.num_faces

        reduce_dim = compute_unit.reduce_dim
        pool_type = compute_unit.reduce_pool

        src_a_reduced_tensor = tilize_block(
            tensor_a, dimensions, output_format, num_faces
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        src_a_reduced_tensor = generate_golden(
            src_a_reduced_tensor,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
        ).flatten()

        dest_golden_tensor = tilize_block(
            tensor_dst, dimensions, output_format, num_faces
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        dest_golden_tensor = generate_golden(
            dest_golden_tensor,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
        ).flatten()

        golden_tensor = torch.zeros(tile_cnt * 1024)

        for i in range(tile_cnt * 1024):
            if compute_unit.reduce_pool == ReducePool.Max:
                golden_tensor[i] = max(src_a_reduced_tensor[i], dest_golden_tensor[i])
            if compute_unit.reduce_pool == ReducePool.Sum:
                golden_tensor[i] = src_a_reduced_tensor[i] + dest_golden_tensor[i]
            if compute_unit.reduce_pool == ReducePool.Average:
                golden_tensor[i] = (
                    src_a_reduced_tensor[i] * 32 + dest_golden_tensor[i]
                ) / 32

        golden_tensor = untilize_block(golden_tensor, output_format, dimensions)

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        pool_type_cpp = compute_unit.reduce_pool.cpp_enum_value
        reduce_dim_cpp = compute_unit.reduce_dim.cpp_enum_value

        return (
            f"// Operation {stage}: Reduce {reduce_dim_cpp} FPU\n"
            f"_llk_math_reduce_init_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, false>();\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        pool_type_cpp = compute_unit.reduce_pool.cpp_enum_value
        reduce_dim_cpp = compute_unit.reduce_dim.cpp_enum_value

        # Create a temporary TensorShape object with Src_A tile dimensions
        tile_shape = operation.src_a.tile_shape
        tensor_shape_instantiation: str = (
            f"ckernel::TensorShape{{{tile_shape.face_r_dim}, {tile_shape.face_c_dim}, {tile_shape.num_faces_r_dim}, {tile_shape.num_faces_c_dim}}}"
        )

        return (
            f"_llk_math_reduce_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, false, false>(\n"
            f"    {block.tile_id_block}, {tensor_shape_instantiation}\n"
            f");\n"
        )
