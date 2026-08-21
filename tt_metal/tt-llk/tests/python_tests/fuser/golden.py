# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from .fpu_node import FpuNode
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation
    from .operand import Operand
    from .pack_node import PackNode
    from .sfpu_node import SfpuNode

from helpers.golden_generators import (
    BinarySFPUGolden,
    BroadcastGolden,
    DataCopyGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    PackGolden,
    ReduceGolden,
    TransposeGolden,
    UnarySFPUGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    EltwiseBinaryReuseDestType,
    ReduceDimension,
    ReducePool,
    Transpose,
)
from helpers.tilize_untilize import tilize_block, untilize_block


class Golden:
    """Golden result helpers shared by compute unit base classes."""

    def tilize_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> torch.Tensor:
        src = node.src_a
        return tilize_block(
            tensor,
            src.dimensions,
            src.data_format,
            src.tile_shape.total_num_faces(),
            tile_dimensions=[
                src.tile_shape.total_row_dim(),
                src.tile_shape.total_col_dim(),
            ],
            face_r_dim=src.tile_shape.face_r_dim,
        )

    def transpose_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        use_srcb: bool = False,
    ) -> torch.Tensor:
        operand = node.src_b if use_srcb else node.src_a
        t_matrix = get_golden_generator(TransposeGolden)
        if node.transpose_faces == Transpose.Yes:
            tensor = t_matrix.transpose_faces_multi_tile(
                tensor,
                config.sentinel.golden_math_format,
                operand.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operand.dimensions,
            )
        if node.transpose_within_face == Transpose.Yes:
            tensor = t_matrix.transpose_within_faces_multi_tile(
                tensor,
                config.sentinel.golden_math_format,
                operand.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operand.dimensions,
            )
        return tensor

    def broadcast_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        operand: "Operand" = None,
        per_block: bool = False,
    ) -> torch.Tensor:
        operand = operand or node.src_b
        if node.broadcast_type == BroadcastType.None_:
            return tensor

        tile_shape = operand.tile_shape
        num_faces = tile_shape.total_num_faces()
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())

        tilized = tilize_block(
            tensor,
            operand.dimensions,
            operand.data_format,
            num_faces,
            tile_dimensions=tile_dims,
        )
        broadcast_generator = get_golden_generator(BroadcastGolden)
        broadcast = broadcast_generator(
            node.broadcast_type,
            tilized,
            operand.data_format,
            num_faces,
            operand.tile_count,
            tile_shape.face_r_dim,
        )

        if per_block:
            tiles = broadcast.view(operand.tile_count_y, operand.tile_count_x, -1)
            for bx in range(0, operand.tile_count_x, operation.block_tiles_x):
                block = tiles[:, bx : bx + operation.block_tiles_x]
                block[:] = block[:, :1]
        elif node.broadcast_tile is not None:
            tiles = broadcast.view(operand.tile_count, -1)
            tiles[:] = tiles[node.broadcast_tile].clone()

        return untilize_block(
            broadcast,
            operand.data_format,
            operand.dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )

    def reuse_dest_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            return None, tensor_a
        return tensor_a, tensor_b

    def eltwise_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        accumulate_on_dest: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        math_fidelity = node.math_fidelity

        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_a = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
            tensor_b = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        generate_golden = get_golden_generator(EltwiseBinaryGolden)
        golden_tensor = generate_golden(
            node.fpu.operation,
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            tile_shape=operation.tile_shape,
        ).reshape(operation.max_output_dimensions)

        if accumulate_on_dest or node.acc_to_dest == AccToDest.Yes:
            golden_tensor = golden_tensor + tensor_dst

        return (tensor_a, tensor_b, golden_tensor)

    def datacopy_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if node.broadcast_type != BroadcastType.None_:
            source_tensor = tensor_b
        else:
            source_tensor = tensor_a

        golden_generator = get_golden_generator(DataCopyGolden)
        golden_tensor = golden_generator(
            source_tensor,
            config.sentinel.golden_math_format,
            num_faces=operation.tile_shape.total_num_faces(),
            input_dimensions=node.src_a.dimensions,
            face_r_dim=operation.tile_shape.face_r_dim,
            tile_shape=operation.tile_shape,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def matmul_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        math_fidelity = node.math_fidelity

        generate_golden = get_golden_generator(MatmulGolden)
        golden = generate_golden(
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            input_A_dimensions=node.src_a.dimensions,
            input_B_dimensions=node.src_b.dimensions,
            tilize=False,
            input_A_format=node.src_a.data_format,
            input_B_format=node.src_b.data_format,
        )

        return (tensor_a, tensor_b, golden)

    def reduce_golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "FpuNode",
        block_max: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        tile_shape = operation.tile_shape
        dimensions = operation.max_output_dimensions
        num_faces = tile_shape.total_num_faces()
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        grid_y, grid_x = dimensions[0] // tile_dims[0], dimensions[1] // tile_dims[1]
        reduce_dim = ReduceDimension.Row if block_max else node.fpu.reduce_dim
        pool_type = ReducePool.Max if block_max else node.fpu.reduce_pool
        pool = torch.amax if pool_type == ReducePool.Max else torch.sum
        generate_golden = get_golden_generator(ReduceGolden)

        def reduce(tensor: torch.Tensor, fold_blocks: bool) -> torch.Tensor:
            reduced = tilize_block(
                tensor, dimensions, output_format, num_faces, tile_dimensions=tile_dims
            ).flatten()
            reduced = generate_golden(
                reduced,
                reduce_dim,
                pool_type,
                output_format,
                tile_cnt=grid_y * grid_x,
                tile_shape=tile_shape,
            ).flatten()

            if fold_blocks:
                tiles = reduced.view(grid_y, grid_x, -1)
                for by in range(0, grid_y, operation.block_tiles_y):
                    for bx in range(0, grid_x, operation.block_tiles_x):
                        block = tiles[
                            by : by + operation.block_tiles_y,
                            bx : bx + operation.block_tiles_x,
                        ]
                        folded = pool(block, dim=1)
                        if not block_max:
                            folded = pool(folded, dim=0, keepdim=True)
                        block[:] = 0
                        block[: len(folded), 0] = folded

            return untilize_block(
                reduced,
                output_format,
                dimensions,
                tile_dimensions=tile_dims,
                num_faces=num_faces,
            ).flatten()

        src_reduced = reduce(tensor_a, block_max or node.reduce_to_tile)
        dest_reduced = reduce(tensor_dst, block_max)

        if pool_type == ReducePool.Average:
            span = tile_dims[1] if reduce_dim == ReduceDimension.Row else tile_dims[0]
            scaler = tensor_b.flatten()[0].item()
            golden_tensor = (src_reduced * span + dest_reduced) * scaler
        else:
            golden_tensor = pool(torch.stack((src_reduced, dest_reduced)), dim=0)

        return (tensor_a, tensor_b, golden_tensor.to(src_reduced.dtype))

    def unary_sfpu_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "SfpuNode",
        batch_dims: tuple,
    ) -> torch.Tensor:
        sfpu = node.sfpu
        format_input = config.sentinel.golden_math_format
        format_output = config.sentinel.golden_math_format
        dest_acc = config.dest_acc

        generate_sfpu_golden = get_golden_generator(UnarySFPUGolden)

        return generate_sfpu_golden(
            sfpu.operation,
            tensor,
            format_output,
            dest_acc,
            format_input,
            batch_dims,
            sfpu.iterations,
            sfpu.dest_idx,
            sfpu.fill_const_value,
            skip_tilize=True,
        )

    def binary_sfpu_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "SfpuNode",
        batch_dims: tuple,
    ) -> torch.Tensor:
        sfpu = node.sfpu
        math_format = config.sentinel.golden_math_format

        generate_binary_golden = get_golden_generator(BinarySFPUGolden)

        return generate_binary_golden(
            sfpu.operation,
            tensor,
            sfpu.dst_index_in0,
            sfpu.dst_index_in1,
            sfpu.dst_index_out,
            sfpu.iterations,
            batch_dims,
            math_format,
            skip_tilize=True,
        )

    def untilize_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "PackNode",
    ) -> torch.Tensor:
        untilize = get_golden_generator(UntilizeGolden)
        tile_shape = node.output.tile_shape
        return untilize(
            tensor,
            node.output.data_format,
            dimensions=node.output.dimensions,
            tile_dimensions=(tile_shape.total_row_dim(), tile_shape.total_col_dim()),
        )

    def l1_acc_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "PackNode",
    ) -> torch.Tensor:
        output_dims = node.output.dimensions
        output_format = node.output.data_format
        tile_size = node.output.tile_shape.total_tile_size()
        tile_count_x = node.output.tile_count_x
        tile_count_y = node.output.tile_count_y
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y

        tile_dims = (
            node.output.tile_shape.total_row_dim(),
            node.output.tile_shape.total_col_dim(),
        )
        num_faces = node.output.tile_shape.total_num_faces()
        tensor = tilize_block(
            tensor,
            output_dims,
            output_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        ).flatten()
        tile_grid = tensor.view(tile_count_y, tile_count_x, tile_size)

        accumulated = torch.zeros(
            block_tiles_y, block_tiles_x, tile_size, dtype=tensor.dtype
        )
        for by in range(0, tile_count_y, block_tiles_y):
            for bx in range(0, tile_count_x, block_tiles_x):
                bty = min(block_tiles_y, tile_count_y - by)
                btx = min(block_tiles_x, tile_count_x - bx)
                accumulated[:bty, :btx] += tile_grid[by : by + bty, bx : bx + btx]

        result_grid = torch.zeros(
            tile_count_y, tile_count_x, tile_size, dtype=tensor.dtype
        )
        result_grid[:block_tiles_y, :block_tiles_x] = accumulated
        return untilize_block(
            result_grid.flatten(),
            output_format,
            output_dims,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )

    def relu_golden(
        self,
        tensor: torch.Tensor,
        config: "GlobalConfig",
        operation: "L1Operation",
        node: "PackNode",
    ) -> torch.Tensor:
        intermediate_format = config.sentinel.golden_pack_src
        relu_config = PackGolden.generate_relu_config(
            node.pack_relu, node.relu_threshold, intermediate_format
        )
        return PackGolden.apply_relu(tensor, relu_config, intermediate_format)
