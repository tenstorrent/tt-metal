# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn.deprecated


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_global_var_toggle_and_device_eps():
    # Check that APIs modifying global vars work
    ttnn.deprecated.device.EnablePersistentKernelCache()
    ttnn.deprecated.device.DisablePersistentKernelCache()
    ttnn.deprecated.device.EnableCompilationReports()
    ttnn.deprecated.device.DisableCompilationReports()
    # Check that the ttnn.deprecated bindings take the correct path
    # to device epsilon constants
    assert ttnn.deprecated.device.EPS_GS == 0.001953125
    assert ttnn.deprecated.device.EPS_WHB0 == 1.1920899822825959e-07
    assert ttnn.deprecated.device.EPS_BH == 1.1920899822825959e-07


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_device_arch():
    assert ttnn.deprecated.device.Arch.GRAYSKULL.name == "GRAYSKULL"
    assert ttnn.deprecated.device.Arch.WORMHOLE_B0.name == "WORMHOLE_B0"
    assert ttnn.deprecated.device.Arch.BLACKHOLE.name == "BLACKHOLE"
    pass
