# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.tilize_untilize import tilize_block, untilize_block

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.llk_params import (
    AccToDest,
    BroadcastType,
    ClearFP32DstAcc,
    DataCopyType,
    EltwiseBinaryReuseDestType,
    EnforceFP32Accumulation,
    MathFidelity,
    PerfRunType,
    Transpose,
    UnpackToDest,
)

from .base_fpu import Fpu
from .base_unpacker import Unpacker
from .block_data import (
    BlockData,
    KernelInvocation,
)
from .operand import Operand


class FpuNode:
    def __init__(
        self,
        fpu: Fpu,
        src_a: Operand,
        src_b: Operand,
        unpacker: Unpacker = None,
        transpose_faces: Transpose = Transpose.No,
        transpose_within_face: Transpose = Transpose.No,
        broadcast_type: BroadcastType = BroadcastType.None_,
        data_copy_type: DataCopyType = DataCopyType.A2D,
        reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE,
        math_fidelity: MathFidelity = MathFidelity.LoFi,
        enforce_fp32_accumulation: EnforceFP32Accumulation = EnforceFP32Accumulation.No,
        clear_fp32_dst_acc: ClearFP32DstAcc = ClearFP32DstAcc.No,
        acc_to_dest: AccToDest = AccToDest.No,
        unpack_to_dest: UnpackToDest = UnpackToDest.No,
        blocks: Optional[List[Tuple[KernelInvocation, ...]]] = None,
        block_defaults: KernelInvocation = KernelInvocation(),
    ):
        self.fpu = fpu
        self.unpacker = unpacker
        self.src_a = src_a
        self.src_b = src_b
        self.transpose_faces = transpose_faces
        self.transpose_within_face = transpose_within_face
        self.broadcast_type = broadcast_type
        self.reuse_dest = reuse_dest
        self.math_fidelity = math_fidelity
        self.enforce_fp32_accumulation = enforce_fp32_accumulation
        self.clear_fp32_dst_acc = clear_fp32_dst_acc
        self.acc_to_dest = acc_to_dest
        self.unpack_to_dest = unpack_to_dest
        self.blocks = blocks
        self.block_defaults = block_defaults

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

    def automatic_call(self, call: KernelInvocation, block: BlockData, unpack=False):
        defaults = (
            replace(self.block_defaults, dest=None) if unpack else self.block_defaults
        )
        values = {
            name: value for name, value in vars(defaults).items() if value is not None
        }
        if self.src_b is None:
            call = replace(call, in1=None)
        return replace(call, **values)

    def unpack_init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None or config.skip_unpack_init:
            return ""
        return self.unpacker.init(operation, config, self, block)

    def unpack_call(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
        call: KernelInvocation,
    ):
        if self.unpacker is None:
            return ""
        if config.perf_run_type == PerfRunType.PACK_ISOLATE:
            return ""
        block.tile_id_global = call.in0
        block.tile_id_src_b = call.in1
        block.tile_id_block = call.dest
        if config.perf_run_type == PerfRunType.MATH_ISOLATE:
            return self.unpacker.perf_set_valid(operation, config, self, block)
        return self.unpacker.unpack(operation, config, self, block)

    def unpack_uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None or config.skip_unpack_init:
            return ""
        return self.unpacker.uninit(operation, config, self, block)

    def fpu_init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.fpu.init(operation, config, self, block)

    def fpu_call(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
        call: KernelInvocation,
    ):
        if config.perf_run_type == PerfRunType.PACK_ISOLATE:
            return ""
        block.tile_id_global = call.in0
        block.tile_id_src_b = call.in1
        block.tile_id_block = call.dest
        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return self.unpacker.perf_clear_valid(operation, config, self, block)
        return self.fpu.calculate(operation, config, self, block)

    def fpu_uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.fpu.uninit(operation, config, self, block)

    def golden(
        self,
        input_tensor_a,
        input_tensor_b,
        tensor_a,
        tensor_b,
        tensor_dst,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.unpacker is not None and self.src_a is not None:
            unpacked_tensor_a, unpacked_tensor_b = self.unpacker.golden(
                input_tensor_a, input_tensor_b, operation, config, self
            )

            if unpacked_tensor_a is not None:
                tensor_a = unpacked_tensor_a

            if unpacked_tensor_b is not None:
                tensor_b = unpacked_tensor_b

        tensor_a, tensor_b, tensor_dst = self.fpu.golden(
            tensor_a, tensor_b, tensor_dst, operation, config, self
        )

        return (
            tensor_a,
            tensor_b,
            tensor_dst.reshape(operation.max_output_dimensions),
        )

    def golden_call(
        self,
        call: KernelInvocation,
        tensor_dst: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
        master: bool,
    ) -> torch.Tensor:
        tile_shape = operation.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        num_faces = tile_shape.total_num_faces()

        def load(operand, index):
            tensor = operand.master_golden if master else operand.raw_data
            tiles = tilize_block(
                tensor,
                operand.dimensions,
                operand.data_format,
                num_faces=operand.tile_shape.total_num_faces(),
                tile_dimensions=(
                    operand.tile_shape.total_row_dim(),
                    operand.tile_shape.total_col_dim(),
                ),
            )
            return untilize_block(
                tiles[index].flatten(),
                operand.data_format,
                tile_dims,
                tile_dimensions=tile_dims,
                num_faces=num_faces,
            )

        tensor_a = load(self.src_a, call.in0)
        tensor_b = (
            load(self.src_b, call.in1)
            if self.src_b is not None
            else torch.zeros(tile_dims)
        )
        current = untilize_block(
            tensor_dst[call.dest].flatten(),
            config.sentinel.golden_math_format,
            tile_dims,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )
        tile_operation = copy(operation)
        tile_operation.max_output_dimensions = tile_dims
        tile_operation.block_size = tile_dims
        tile_operation.block_tiles_x = 1
        tile_operation.block_tiles_y = 1
        if self.fpu.block_operation == "Datacopy":
            result = get_golden_generator(DataCopyGolden)(
                tensor_a,
                config.sentinel.golden_math_format,
                num_faces=num_faces,
                input_dimensions=tile_dims,
                face_r_dim=tile_shape.face_r_dim,
                tile_shape=tile_shape,
            )
        else:
            _, _, result = self.fpu.golden(
                tensor_a, tensor_b, current, tile_operation, config, self
            )
        tensor_dst[call.dest] = tilize_block(
            result,
            tile_dims,
            config.sentinel.golden_math_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        )[0]
        return tensor_dst

    def __str__(self):
        unpacker = (
            f"{type(self.unpacker).__name__}" if self.unpacker is not None else ""
        )
        return f"{unpacker}, {self.fpu}, {self.math_fidelity}"
