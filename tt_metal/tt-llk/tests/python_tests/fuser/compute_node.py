# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from helpers.llk_params import (
    BroadcastType,
    DataCopyType,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    PerfRunType,
    ReduceDimension,
    ReducePool,
    Transpose,
)
from helpers.tilize_untilize import tilize_block, untilize_block

from .block_data import BlockData
from .fused_fpu import Fpu
from .fused_sfpu import Sfpu
from .fused_unpacker import Unpacker


class ComputeNode:
    def __init__(
        self,
        unpacker: Unpacker = None,
        fpu: Fpu = None,
        sfpu: Sfpu = None,
        unpack_transpose_faces: Transpose = Transpose.No,
        unpack_transpose_within_face: Transpose = Transpose.No,
        broadcast_type: BroadcastType = BroadcastType.None_,
        data_copy_type: DataCopyType = DataCopyType.A2D,
        reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE,
        reduce_dim: ReduceDimension = None,
        reduce_pool: ReducePool = ReducePool.Max,
        math_fidelity: MathFidelity = MathFidelity.LoFi,
    ):
        if fpu is None and sfpu is None:
            raise ValueError("Compute unit needs an fpu or sfpu unit")
        if fpu is not None and sfpu is not None:
            raise ValueError("Compute unit can be only fpu or sfpu")
        if sfpu is not None and unpacker is not None:
            raise ValueError("Sfpu unit does not support unpacker")

        self.unpacker = unpacker
        self.fpu = fpu
        self.sfpu = sfpu
        self.unpack_transpose_faces = unpack_transpose_faces
        self.unpack_transpose_within_face = unpack_transpose_within_face
        self.broadcast_type = broadcast_type
        self.reuse_dest = reuse_dest
        self.reduce_dim = reduce_dim
        self.reduce_pool = reduce_pool
        self.math_fidelity = math_fidelity

        if (
            self.broadcast_type != BroadcastType.None_
            and data_copy_type == DataCopyType.A2D
        ):
            self.data_copy_type = DataCopyType.B2D
        elif (
            self.broadcast_type == BroadcastType.None_
            and data_copy_type == DataCopyType.B2D
        ):
            self.data_copy_type = DataCopyType.A2D
        else:
            self.data_copy_type = data_copy_type

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None:
            return ""

        code = ""
        skip_init = config.perf_run_type in (
            PerfRunType.PACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        )
        if not skip_init:
            code += self.unpacker().init(operation, config, self, block)

        code += self.unpacker().loop.unpack_loop(operation, config, self, block)
        if not skip_init:
            code += self.unpacker().uninit(operation, config, self, block)

        return code

    def fpu_calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.fpu is None:
            return ""

        code = ""
        skip_init = config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        )
        if not skip_init:
            code += self.fpu.init(operation, config, self, block)

        code += self.fpu.loop.math_loop(operation, config, self, block)
        if not skip_init:
            code += self.fpu.uninit(operation, config, self, block)

        return code

    def sfpu_calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.sfpu is None:
            return ""

        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        code = self.sfpu.init(operation, config, self, block)
        code += self.sfpu.calculate(operation, config, self, block)
        code += self.sfpu.uninit(operation, config, self, block)
        return code

    def math_calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        if self.fpu is not None:
            return self.fpu_calculate(operation, config, block)
        elif self.sfpu is not None:
            return self.sfpu_calculate(operation, config, block)
        else:
            raise ValueError("fpu and sfpu are not defined")

    def golden(
        self,
        input_tensor_a,
        input_tensor_b,
        tensor_a,
        tensor_b,
        tensor_dst,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.unpacker is not None:
            unpacked_tensor_a, unpacked_tensor_b = self.unpacker().golden(
                input_tensor_a, input_tensor_b, operation, config, self
            )

            if unpacked_tensor_a is not None:
                tensor_a = unpacked_tensor_a

            if unpacked_tensor_b is not None:
                tensor_b = unpacked_tensor_b

        if self.fpu is not None:
            tensor_a, tensor_b, tensor_dst = self.fpu.golden(
                tensor_a, tensor_b, tensor_dst, operation, config, self
            )

        if self.sfpu is not None:
            tilized_dst = tilize_block(
                tensor_dst,
                operation.max_output_dimensions,
                operation.output.data_format,
            )

            tile_count_x = operation.output.tile_count_x
            tile_count_y = operation.output.tile_count_y
            block_tiles_x = operation.block_tiles_x
            block_tiles_y = operation.block_tiles_y

            full_blocks_x = tile_count_x // block_tiles_x
            full_blocks_y = tile_count_y // block_tiles_y
            remaining_tiles_x = tile_count_x % block_tiles_x
            remaining_tiles_y = tile_count_y % block_tiles_y

            full_x_limit = full_blocks_x * block_tiles_x
            full_y_limit = full_blocks_y * block_tiles_y

            tile_size = tilized_dst.shape[1]

            def process_block(block_x, block_y, block_tiles_x_eff, block_tiles_y_eff):
                block_tile_ids = []
                for tile_y in range(block_tiles_y_eff):
                    for tile_x in range(block_tiles_x_eff):
                        tile_id = tile_count_x * (block_y + tile_y) + (block_x + tile_x)
                        block_tile_ids.append(tile_id)

                block_tile_cnt = len(block_tile_ids)
                if block_tile_cnt == 0:
                    return

                block_tensor = tilized_dst[block_tile_ids, :].clone().flatten()
                block_dims = (block_tile_cnt * 32, 32)

                block_tensor = self.sfpu.golden(
                    block_tensor,
                    operation,
                    config,
                    self,
                    block_dims,
                    block_tile_cnt,
                )

                tilized_dst[block_tile_ids, :] = block_tensor.view(
                    block_tile_cnt, tile_size
                )

            if full_blocks_x > 0 and full_blocks_y > 0:
                for block_x in range(0, full_x_limit, block_tiles_x):
                    for block_y in range(0, full_y_limit, block_tiles_y):
                        process_block(block_x, block_y, block_tiles_x, block_tiles_y)

            if remaining_tiles_y > 0 and full_blocks_x > 0:
                for block_x in range(0, full_x_limit, block_tiles_x):
                    process_block(
                        block_x, full_y_limit, block_tiles_x, remaining_tiles_y
                    )

            if remaining_tiles_x > 0 and full_blocks_y > 0:
                for block_y in range(0, full_y_limit, block_tiles_y):
                    process_block(
                        full_x_limit, block_y, remaining_tiles_x, block_tiles_y
                    )

            if remaining_tiles_x > 0 and remaining_tiles_y > 0:
                process_block(
                    full_x_limit, full_y_limit, remaining_tiles_x, remaining_tiles_y
                )

            tensor_dst = untilize_block(
                tilized_dst.flatten(),
                operation.output.data_format,
                operation.max_output_dimensions,
            ).reshape(operation.max_output_dimensions)

        return (
            tensor_a.reshape(operation.src_a.dimensions),
            tensor_b.reshape(operation.src_b.dimensions),
            tensor_dst.reshape(operation.max_output_dimensions),
        )

    def __str__(self):
        if self.fpu is not None:
            unpacker = f"{self.unpacker.__name__}" if self.unpacker is not None else ""
            return f"{unpacker}, {self.fpu}, {self.math_fidelity}"
        elif self.sfpu:
            return f"{self.sfpu}"
        else:
            return ""
