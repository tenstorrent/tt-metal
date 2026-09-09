# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.fpu.datacopy import datacopy_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation


class UnaryBroadcastFpu(Fpu):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(datacopy_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_unary_broadcast.h",
        ]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        tensor_shape = operation.tile_shape.cpp_value
        return (
            f"// Operation {stage}: Unary Broadcast FPU\n"
            f"_llk_math_eltwise_unary_broadcast_init_<{broadcast_type}, false>"
            f"({tensor_shape});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_math_eltwise_unary_broadcast_({block.tile_id_block});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
