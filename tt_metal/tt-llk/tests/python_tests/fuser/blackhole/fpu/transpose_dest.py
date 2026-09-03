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


class TransposeDestFpu(Fpu):
    granularity = InvocationGranularity.TILE

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_transpose_dest.h",
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
        golden_tensor = self.transpose_golden(
            tensor_dst, config, operation, compute_unit
        )
        return tensor_a, tensor_b, golden_tensor

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        is_32bit = config.dest_acc.cpp_enum_value
        transpose_faces = compute_unit.transpose_faces.cpp_enum_value
        return f"_llk_math_transpose_dest_init_<{transpose_faces}, {is_32bit}>();\n"

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        is_32bit = config.dest_acc.cpp_enum_value
        transpose_faces = compute_unit.transpose_faces.cpp_enum_value
        return f"_llk_math_transpose_dest_<{transpose_faces}, {is_32bit}>({block.tile_id_block});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_transpose_dest_uninit_();\n"
