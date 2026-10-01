# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.base_sfpu import Sfpu
from fuser.golden.sfpu.topk import topk_golden
from fuser.indexing import InvocationGranularity
from helpers.llk_params import MathOperation


class TopKSfpu(Sfpu):
    granularity = InvocationGranularity.BLOCK
    golden_fn = staticmethod(topk_golden)

    def __init__(self, operation, k=32, descending=True, m_iter=0, skip_second=False):
        self.operation = operation
        self.k = k
        self.descending = descending
        self.m_iter = m_iter
        self.skip_second = skip_second

    def get_headers(self):
        return [
            "llk_sfpu/ckernel_sfpu_topk.h",
            "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h",
        ]

    def init(self, operation, config, compute_unit, block):
        return (
            "SFPU_UNARY_INIT_FN(topk_local_sort, ckernel::sfpu::topk_init, (false));\n"
        )

    def calculate(self, operation, config, compute_unit, block):
        direction = int(not self.descending)
        logk = self.k.bit_length() - 1
        templates = f"false, {config.dest_acc.cpp_enum_value}"
        if self.operation == MathOperation.TopKLocalSort:
            function = "calculate_bitonic_topk_phases_steps"
            args = f"{direction}, {logk - 1}, 0, 10, 0"
        elif self.operation == MathOperation.TopKMerge:
            function = "calculate_bitonic_topk_merge"
            templates += f", {direction}"
            args = f"{self.m_iter}, {self.k}"
        else:
            function = "calculate_bitonic_topk_rebuild"
            args = (
                f"{direction}, {self.m_iter}, {self.k}, {logk}, {int(self.skip_second)}"
            )
        return (
            f"SFPU_UNARY_CALL({operation.dest_sync.cpp_enum_value}, "
            f"{config.dest_acc.cpp_enum_value}, {function}, ({templates}), "
            f"{block.tile_id_dest}, VectorMode::RC_custom, {args});\n"
        )

    def __str__(self):
        return f"TopKSfpu({self.operation}, k={self.k})"
