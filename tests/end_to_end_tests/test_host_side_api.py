# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_global_var_toggle_and_device_eps():
    # Check that APIs modifying global vars work
    ttnn.device.EnablePersistentKernelCache()
    ttnn.device.DisablePersistentKernelCache()
    ttnn.device.EnableCompilationReports()
    ttnn.device.DisableCompilationReports()
    # Check that the tt_lib bindings take the correct path
    # to device epsilon constants
    assert ttnn.device.EPS_GS == 0.001953125
    assert ttnn.device.EPS_WHB0 == 1.1920899822825959e-07
    assert ttnn.device.EPS_BH == 1.1920899822825959e-07


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_device_arch():
    assert ttnn.device.Arch.GRAYSKULL.name == "GRAYSKULL"
    assert ttnn.device.Arch.WORMHOLE_B0.name == "WORMHOLE_B0"
    assert ttnn.device.Arch.BLACKHOLE.name == "BLACKHOLE"
    pass
