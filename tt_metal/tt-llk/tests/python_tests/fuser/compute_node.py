# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    ClearFP32DstAcc,
    DataCopyType,
    EltwiseBinaryReuseDestType,
    EnforceFP32Accumulation,
    MathFidelity,
    Transpose,
    UnpackToDest,
)
from helpers.tilize_untilize import tilize_block, untilize_block

from .block_data import BlockData
from .fused_fpu import Fpu
from .fused_operand import Operand
from .fused_sfpu import Sfpu
from .fused_unpacker import Unpacker


class ComputeNode:
    def __init__(
        self,
        unpacker: Unpacker = None,
        fpu: Fpu = None,
        sfpu: Sfpu = None,
        src_a: Operand = None,
        src_b: Operand = None,
        unpack_transpose_faces: Transpose = Transpose.No,
        unpack_transpose_within_face: Transpose = Transpose.No,
        broadcast_type: BroadcastType = BroadcastType.None_,
        data_copy_type: DataCopyType = DataCopyType.A2D,
        reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE,
        math_fidelity: MathFidelity = MathFidelity.LoFi,
        enforce_fp32_accumulation: EnforceFP32Accumulation = EnforceFP32Accumulation.No,
        clear_fp32_dst_acc: ClearFP32DstAcc = ClearFP32DstAcc.No,
        acc_to_dest: AccToDest = AccToDest.No,
        unpack_to_dest: UnpackToDest = UnpackToDest.No,
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
        self.math_fidelity = math_fidelity
        self.enforce_fp32_accumulation = enforce_fp32_accumulation
        self.clear_fp32_dst_acc = clear_fp32_dst_acc
        self.acc_to_dest = acc_to_dest
        self.unpack_to_dest = unpack_to_dest
        self.src_a = src_a
        self.src_b = src_b

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

        if self.src_a is None and self.src_b is None:
            return

    def unpack_reconfig(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ):
        if self.unpacker is None or config.skip_unpack_init:
            return ""
        return config.sentinel.configure_unpack(config, operation, self)

    def unpack_init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None or config.skip_unpack_init:
            return ""
        return self.unpacker.init(operation, config, self, block)

    def unpack_run(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None:
            return ""
        return self.unpacker.loop.unpack_loop(operation, config, self, block)

    def unpack_uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None or config.skip_unpack_init:
            return ""
        return self.unpacker.uninit(operation, config, self, block)

    def math_reconfig(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ):
        if config.skip_math_init or self.fpu is None:
            return ""
        return config.sentinel.configure_math(config, operation, self)

    def math_init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        if self.fpu is not None:
            return self.fpu.init(operation, config, self, block)
        elif self.sfpu is not None:
            return self.sfpu.init(operation, config, self, block)
        return ""

    def math_run(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.fpu is not None:
            return self.fpu.loop.math_loop(operation, config, self, block)
        elif self.sfpu is not None:
            if config.skip_math_init:
                return ""
            return self.sfpu.calculate(operation, config, self, block)
        return ""

    def math_uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        if self.fpu is not None:
            return self.fpu.uninit(operation, config, self, block)
        elif self.sfpu is not None:
            return self.sfpu.uninit(operation, config, self, block)
        return ""

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
            unpacked_tensor_a, unpacked_tensor_b = self.unpacker.golden(
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
            tile_dims = (
                operation.tile_shape.total_row_dim(),
                operation.tile_shape.total_col_dim(),
            )
            num_faces = operation.tile_shape.total_num_faces()
            tilized_dst = tilize_block(
                tensor_dst,
                operation.max_output_dimensions,
                config.sentinel.golden_math_format,
                num_faces=num_faces,
                tile_dimensions=tile_dims,
            )

            tile_count_x = (
                operation.max_output_dimensions[1]
                // operation.tile_shape.total_col_dim()
            )
            tile_count_y = (
                operation.max_output_dimensions[0]
                // operation.tile_shape.total_row_dim()
            )
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
                block_dims = (
                    block_tile_cnt * operation.tile_shape.total_row_dim(),
                    operation.tile_shape.total_col_dim(),
                )

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
                config.sentinel.golden_math_format,
                operation.max_output_dimensions,
                tile_dimensions=tile_dims,
                num_faces=num_faces,
            ).reshape(operation.max_output_dimensions)

        return (
            tensor_a,
            tensor_b,
            tensor_dst.reshape(operation.max_output_dimensions),
        )

    def __str__(self):
        if self.fpu is not None:
            unpacker = (
                f"{type(self.unpacker).__name__}" if self.unpacker is not None else ""
            )
            return f"{unpacker}, {self.fpu}, {self.math_fidelity}"
        elif self.sfpu:
            return f"{self.sfpu}"
        else:
            return ""
