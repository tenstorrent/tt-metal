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

from .block_data import BlockData
from .fused_fpu import Fpu
from .fused_operand import Operand
from .fused_unpacker import Unpacker


class FpuNode:
    def __init__(
        self,
        fpu: Fpu,
        src_a: Operand,
        src_b: Operand,
        unpacker: Unpacker = None,
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
        reduce_to_tile: bool = False,
    ):
        self.fpu = fpu
        self.unpacker = unpacker
        self.src_a = src_a
        self.src_b = src_b
        self.unpack_transpose_faces = unpack_transpose_faces
        self.unpack_transpose_within_face = unpack_transpose_within_face
        self.broadcast_type = broadcast_type
        self.reuse_dest = reuse_dest
        self.math_fidelity = math_fidelity
        self.enforce_fp32_accumulation = enforce_fp32_accumulation
        self.clear_fp32_dst_acc = clear_fp32_dst_acc
        self.acc_to_dest = acc_to_dest
        self.unpack_to_dest = unpack_to_dest
        self.reduce_to_tile = reduce_to_tile

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
        config.sentinel.ensure_unpack_buf_desc_ids(self)
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
        if config.skip_math_init:
            return ""
        return config.sentinel.configure_math(config, operation, self)

    def fpu_init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.fpu.init(operation, config, self, block)

    def fpu_run(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        return self.fpu.loop.math_loop(operation, config, self, block)

    def fpu_uninit(
        self,
        operation: "FusedOperation",
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

        tensor_a, tensor_b, tensor_dst = self.fpu.golden(
            tensor_a, tensor_b, tensor_dst, operation, config, self
        )

        return (
            tensor_a,
            tensor_b,
            tensor_dst.reshape(operation.max_output_dimensions),
        )

    def __str__(self):
        unpacker = (
            f"{type(self.unpacker).__name__}" if self.unpacker is not None else ""
        )
        return f"{unpacker}, {self.fpu}, {self.math_fidelity}"
