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
from helpers.llk_params import MathOperation


class EltwiseFpu(Fpu):
    loop: TileLoop = LoopTileByTile()

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
            tensor_a,
            tensor_b,
            tensor_dst,
            config,
            operation,
            compute_unit,
            accumulate_on_dest=self.operation == MathOperation.Elwmul,
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
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value

        return (
            f"// Operation {stage}: Eltwise {op} FPU\n"
            f"_llk_math_eltwise_binary_init_<ckernel::EltwiseBinaryType::{op}, {broadcast_type}, {math_fidelity}, {reuse_dest}>"
            f"({tensor_shape}, {acc_to_dest});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        dest_sync = operation.dest_sync.cpp_enum_value
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        op = self.operation.cpp_enum_value
        tensor_shape = operation.tile_shape.cpp_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        clear_fp32_dst_acc = compute_unit.clear_fp32_dst_acc.cpp_enum_value

        return (
            f"_llk_math_eltwise_binary_<ckernel::EltwiseBinaryType::{op}, {broadcast_type}, {dest_sync},\n"
            f"{dest_acc}, {math_fidelity}, {reuse_dest}>"
            f"({tensor_shape}, {block.tile_id_block}, {clear_fp32_dst_acc});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_eltwise_binary_uninit_();\n"

    def __str__(self) -> str:
        return f"EltwiseFpu({self.operation})"
