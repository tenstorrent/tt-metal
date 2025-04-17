# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.conv2d_common import run_conv2d_short_sweep
from tests.sweep_framework.sweeps.conv2d.short.conv2d_short_sweep import parameters as parameters_ttnn_pytorch
from tests.sweep_framework.sweeps.conv2d.short.conv2d_short_sweep import (
    failing_parameters as failing_parameters_ttnn_pytorch,
)

from tests.sweep_framework.sweeps.conv2d.short.conv2d_ttforge_sweep import parameters as parameters_ttnn_forge
from tests.sweep_framework.sweeps.conv2d.short.conv2d_ttforge_sweep import (
    failing_parameters as failing_parameters_ttnn_forge,
)

from models.utility_functions import (
    skip_for_grayskull,
    is_wormhole_b0,
)

import pytest


@skip_for_grayskull()
@pytest.mark.parametrize("input_spec", parameters_ttnn_pytorch["short_sweep_suite_conv2d"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_pytorch_sweep(device, input_spec):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    # Check if input_spec is in failing_parameters
    if input_spec in failing_parameters_ttnn_pytorch:
        pytest.skip(f"Skipping test for failing input_spec: {input_spec}")

    pcc, messsage = run_conv2d_short_sweep(
        input_spec,
        device,
    )[0]
    assert pcc, messsage


@skip_for_grayskull()
@pytest.mark.parametrize("input_spec", parameters_ttnn_forge["ttforge_sweep_conv2d"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_tt_forge_sweep(device, input_spec):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    # Check if input_spec is in failing_parameters
    if input_spec in failing_parameters_ttnn_forge:
        pytest.skip(f"Skipping test for failing input_spec: {input_spec}")

    pcc, messsage = run_conv2d_short_sweep(
        input_spec,
        device,
    )[0]
    assert pcc, messsage
