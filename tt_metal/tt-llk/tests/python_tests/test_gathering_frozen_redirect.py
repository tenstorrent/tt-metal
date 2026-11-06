# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression: a stale branch prediction at ckernel::disable_gathering() must not hang the TRISC.

See sources/gathering_frozen_redirect_test.cpp. Without the fix in disable_gathering() the test hangs.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter

pytestmark = [skip_for_wormhole, skip_for_quasar]


@dataclass
class GFR_LEAD(TemplateParameter):
    n: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define GFR_LEAD {self.n}"


# Dependent L1 loads just before the bracket: they stall execute so fetch gets ahead of it. Without the
# fix in disable_gathering() every count here hangs.
@pytest.mark.parametrize("lead", [1, 4, 16])
def test_gathering_frozen_redirect(lead):
    if not TestConfig.gathering_build():
        pytest.skip("needs instruction gathering on (TT_METAL_ENABLE_GATHERING=1)")
    src = torch.zeros(1024, dtype=torch.bfloat16)
    config = TestConfig(
        "sources/gathering_frozen_redirect_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        templates=[GFR_LEAD(lead)],
        variant_stimuli=StimuliConfig(
            src,
            DataFormat.Float16_b,
            src,
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
    )
    config.run()
