# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData, InvocationGranularity
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from helpers.llk_params import MathOperation

from .eltwise import EltwiseFpu


class SubBcastColCustomFpu(EltwiseFpu):
    granularity = InvocationGranularity.ROW
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
        operation: L1Operation,
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        return f"_llk_math_sub_bcast_cols_reuse_custom_({ct_dim});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_eltwise_binary_uninit_custom_();\n"
