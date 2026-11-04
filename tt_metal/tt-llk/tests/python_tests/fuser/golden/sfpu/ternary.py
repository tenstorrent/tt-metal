# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import WhereGolden, get_golden_generator
from helpers.llk_params import MathOperation

TERNARY_GOLDENS = {
    MathOperation.SfpuWhere: WhereGolden,
}


def ternary_golden(call, state, node, operation, config):
    golden_cls = TERNARY_GOLDENS.get(node.sfpu.operation)
    if golden_cls is None:
        raise ValueError(f"No ternary SFPU golden for {node.sfpu.operation}")
    result = get_golden_generator(golden_cls)(
        state.dest.get(call.src0),
        state.dest.get(call.src1),
        state.dest.get(call.src2),
    )
    state.dest.set(call.dest, result)
