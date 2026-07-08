# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_loop import FusedLoop, LoopBlockRow
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.llk_params import MathOperation

from .eltwise import EltwiseFpu


class SubBcastColCustomFpu(EltwiseFpu):
    loop: FusedLoop = LoopBlockRow()
    per_block_init = True

    def __init__(self):
        super().__init__(MathOperation.Elwsub)

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "experimental/llk_math_eltwise_binary_custom.h",
        ]

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        num_faces = operation.tile_shape.total_num_faces()
        return (
            f"// Operation {stage}: SubBcastColCustom FPU\n"
            f"_llk_math_eltwise_binary_init_custom_<ckernel::EltwiseBinaryType::ELWSUB, ckernel::BroadcastType::COL>({num_faces});\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        tile_shape = operation.tile_shape
        tensor_shape_instantiation = f"ckernel::TensorShape{{{tile_shape.face_r_dim}, {tile_shape.face_c_dim}, {tile_shape.num_faces_r_dim}, {tile_shape.num_faces_c_dim}}}"
        return f"_llk_math_sub_bcast_cols_reuse_custom_({ct_dim}, {tensor_shape_instantiation}, {block.tile_id_block});\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_eltwise_binary_uninit_custom_();\n"
