# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.fpu.transpose_dest import transpose_dest_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.quasar.fpu.common import configure_alu_cfg


class TransposeDestFpu(Fpu):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(transpose_dest_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_transpose_dest.h",
        ]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        en_32bit_dest = config.dest_acc.cpp_enum_value
        transpose_faces = compute_unit.transpose_faces.cpp_enum_value
        return configure_alu_cfg(
            config, cfg_type="AluCfgType::MOV_OPS_EXPLICIT_FORMAT"
        ) + (f"_llk_math_transpose_dest_init_<{transpose_faces}, {en_32bit_dest}>();\n")

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_math_transpose_dest_({block.tile_id_dest});\n"
