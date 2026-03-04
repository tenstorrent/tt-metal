# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.conv_transpose2d_common import run_short
from tests.sweep_framework.sweeps.conv_transpose2d.short.conv_transpose2d_short_sweep import (
    parameters as parameters_conv_transpose2d,
)

from models.common.utility_functions import (
    is_wormhole_b0,
)

import pytest


@pytest.mark.parametrize("input_spec", parameters_conv_transpose2d["short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_conv_transpose2d_sweep(device, input_spec):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    passed, pcc, *_ = run_short(input_spec, device)
    print(pcc)
    assert passed, pcc
    assert pcc != 1, "conv_transpose2d with randomized input and weights can't legitimately return PCC of 1"
