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
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import AccToDest, EltwiseBinaryReuseDestType, MathOperation


class EltwiseFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format
        math_fidelity = compute_unit.math_fidelity

        if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_a = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
            tensor_b = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        generate_golden = get_golden_generator(EltwiseBinaryGolden)
        golden_tensor = generate_golden(
            self.operation, tensor_a, tensor_b, output_format, math_fidelity
        ).reshape(operation.max_output_dimensions)

        if compute_unit.acc_to_dest == AccToDest.Yes:
            golden_tensor = golden_tensor + tensor_dst

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
        op = self.operation.cpp_enum_value
        face_r_dim = operation.output.tile_shape.face_r_dim
        face_c_dim = operation.output.tile_shape.face_c_dim
        num_faces_r_dim = compute_unit.src_a.tile_shape.total_row_dim() // face_r_dim
        num_faces_c_dim = compute_unit.src_a.tile_shape.total_col_dim() // face_c_dim
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value

        return (
            f"// Operation {stage}: Eltwise {op} FPU\n"
            f"_llk_math_eltwise_binary_init_<ckernel::EltwiseBinaryType::{op}, {broadcast_type}, {math_fidelity}, {reuse_dest}>"
            f"(ckernel::TensorShape{{{face_r_dim}, {face_c_dim}, {num_faces_r_dim}, {num_faces_c_dim}}}, {acc_to_dest});\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        op = self.operation.cpp_enum_value
        face_r_dim = operation.output.tile_shape.face_r_dim
        face_c_dim = operation.output.tile_shape.face_c_dim
        num_faces_r_dim = compute_unit.src_a.tile_shape.total_row_dim() // face_r_dim
        num_faces_c_dim = compute_unit.src_a.tile_shape.total_col_dim() // face_c_dim
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        clear_fp32_dst_acc = compute_unit.clear_fp32_dst_acc.cpp_enum_value

        return (
            f"_llk_math_eltwise_binary_<{op}, {broadcast_type}, dest_sync{stage},\n"
            f"{dest_acc}, {math_fidelity}, {reuse_dest}>"
            f"(ckernel::TensorShape{{{face_r_dim}, {face_c_dim}, {num_faces_r_dim}, {num_faces_c_dim}}}, {block.tile_id_block}, {clear_fp32_dst_acc}\n"
            f");\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_eltwise_binary_uninit_();\n"

    def __str__(self) -> str:
        return f"EltwiseFpu({self.operation})"
