# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_fpu import Fpu
from fuser.block_data import BlockData, InvocationGranularity
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from helpers.llk_params import BroadcastType, MathOperation


class EltwiseFpu(Fpu):
    granularity = InvocationGranularity.TILE

    def __init__(self, operation: MathOperation):
        if not operation in MathOperation.get_fpu_binary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid FPU binary operation."
            )
        self.operation = operation

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_binary.h",
            "llk_math_eltwise_binary_broadcast.h",
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
        return self.eltwise_golden(
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
        op = self.operation.cpp_enum_value
        tensor_shape = operation.tile_shape.cpp_value

        if compute_unit.broadcast_type != BroadcastType.None_:
            broadcast_type = compute_unit.broadcast_type.cpp_enum_value
            return (
                f"// Operation {stage}: Eltwise {op} broadcast FPU\n"
                f"_llk_math_eltwise_binary_broadcast_init_<ckernel::EltwiseBinaryType::{op}, {broadcast_type}, {math_fidelity}>"
                f"({tensor_shape});\n"
            )

        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value

        return (
            f"// Operation {stage}: Eltwise {op} FPU\n"
            f"_llk_math_eltwise_binary_init_<ckernel::EltwiseBinaryType::{op}, {math_fidelity}, {reuse_dest}>"
            f"({tensor_shape}, {acc_to_dest});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        op = self.operation.cpp_enum_value

        if compute_unit.broadcast_type != BroadcastType.None_:
            return f"_llk_math_eltwise_binary_broadcast_({block.tile_id_block});\n"

        tensor_shape = operation.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        clear_fp32_dst_acc = compute_unit.clear_fp32_dst_acc.cpp_enum_value

        return (
            f"_llk_math_eltwise_binary_<ckernel::EltwiseBinaryType::{op}, {reuse_dest}>"
            f"({block.tile_id_block}, {tensor_shape}, {clear_fp32_dst_acc});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""

    def __str__(self) -> str:
        return f"EltwiseFpu({self.operation})"
