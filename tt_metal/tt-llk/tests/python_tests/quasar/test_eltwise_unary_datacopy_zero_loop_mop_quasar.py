# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for tt-metal#60241 item 4 / #56825: zero-length math MOPs.
# See sources/quasar/eltwise_unary_datacopy_zero_loop_mop_quasar_test.cpp for the modes.
# Run one variant at a time on the simulator: a hang can leave it wedged for the next one.
from dataclasses import dataclass

import pytest

# Module import, not `from ... import test_*`, so pytest does not collect the base test here too.
import quasar.test_eltwise_unary_datacopy_quasar as base
from helpers.format_config import DataFormat
from helpers.llk_params import DataCopyType, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.test_variant_parameters import TemplateParameter

# One plain case: Float16_b, A2D, 32x32 tiles, first dest sync / dims / dest index.
COMBINATION = next(
    c
    for c in base.generate_eltwise_unary_datacopy_combinations(
        input_output_formats([DataFormat.Float16_b]), tile_sizes=[(32, 32)]
    )
    if c[2] == DataCopyType.A2D
)


# name -> (mode, count). Plain names so `-k` can select them (it rejects commas).
ZERO_LOOP_MOP_VARIANTS = {
    "baseline": (0, 0),
    "outer0_x1": (1, 1),
    "outer0_x32": (1, 32),
    "inner0_nop_x1": (2, 1),
    "inner0_nop_x32": (2, 32),
    "inner0_startop_x1": (3, 1),
}


@dataclass
class ZERO_LOOP_MOP(TemplateParameter):
    mode: int = 0
    count: int = 0

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int ZERO_LOOP_MOP = {self.mode};\n"
            f"constexpr std::uint32_t ZERO_LOOP_MOP_COUNT = {self.count};"
        )


@pytest.mark.quasar
@parametrize(
    # Passed through parametrize so it unwraps the runtime() markers in the combination.
    combination=[COMBINATION],
    zero_loop_mop=list(ZERO_LOOP_MOP_VARIANTS),
)
def test_eltwise_unary_datacopy_zero_loop_mop_quasar(combination, zero_loop_mop):
    mode, count = ZERO_LOOP_MOP_VARIANTS[zero_loop_mop]
    base.test_eltwise_unary_datacopy_quasar(
        combination,
        [PerfRunType.L1_TO_L1],
        1,
        test_name="sources/quasar/eltwise_unary_datacopy_zero_loop_mop_quasar_test.cpp",
        extra_templates=(ZERO_LOOP_MOP(mode, count),),
    )
