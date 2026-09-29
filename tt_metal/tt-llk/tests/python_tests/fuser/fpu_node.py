# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

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
from .block_data import BlockData
from .indexing import KernelInvocation
from .operand import Operand


class FpuNode:
    block_tiles_x = None
    block_tiles_y = None

    def __init__(
        self,
        fpu: Fpu,
        src_a: Operand,
        src_b: Operand,
        unpacker: Unpacker = None,
        transpose_faces: Transpose = Transpose.No,
        transpose_within_face: Transpose = Transpose.No,
        broadcast_type: BroadcastType = BroadcastType.None_,
        reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE,
        math_fidelity: MathFidelity = MathFidelity.LoFi,
        enforce_fp32_accumulation: EnforceFP32Accumulation = EnforceFP32Accumulation.No,
        acc_to_dest: AccToDest = AccToDest.No,
        unpack_to_dest: UnpackToDest = UnpackToDest.No,
        index_spec=None,
    ):
        self.fpu = fpu
        self.unpacker = unpacker
        self.index_spec = index_spec
        self.src_a = src_a
        self.src_b = src_b
        self.transpose_faces = transpose_faces
        self.transpose_within_face = transpose_within_face
        self.broadcast_type = broadcast_type
        self.reuse_dest = reuse_dest
        self.math_fidelity = math_fidelity
        self.enforce_fp32_accumulation = enforce_fp32_accumulation
        self.acc_to_dest = acc_to_dest
        self.unpack_to_dest = unpack_to_dest

    @property
    def data_copy_type(self) -> DataCopyType:
        return (
            DataCopyType.B2D
            if self.broadcast_type != BroadcastType.None_
            else DataCopyType.A2D
        )

    @property
    def clear_fp32_dst_acc(self) -> ClearFP32DstAcc:
        if self.reuse_dest in (
            EltwiseBinaryReuseDestType.DEST_TO_SRCA,
            EltwiseBinaryReuseDestType.DEST_TO_SRCB,
        ):
            return ClearFP32DstAcc.Yes
        return ClearFP32DstAcc.No

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
    ) -> str:
        if self.unpacker is None or config.perf_run_type == PerfRunType.PACK_ISOLATE:
            return ""
        block.tile_id_src_a = call.in0
        block.tile_id_src_b = call.in1
        block.tile_id_dest = call.dest
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
    ) -> str:
        if config.perf_run_type == PerfRunType.PACK_ISOLATE:
            return ""
        block.tile_id_src_a = call.in0
        block.tile_id_src_b = call.in1
        block.tile_id_dest = call.dest
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

    def __str__(self):
        unpacker = (
            f"{type(self.unpacker).__name__}" if self.unpacker is not None else ""
        )
        return f"{unpacker}, {self.fpu}, {self.math_fidelity}"
