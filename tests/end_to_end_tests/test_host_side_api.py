# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import tt_lib


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_global_var_toggle_and_device_eps():
    # Check that APIs modifying global vars work
    tt_lib.device.EnablePersistentKernelCache()
    tt_lib.device.DisablePersistentKernelCache()
    tt_lib.device.EnableCompilationReports()
    tt_lib.device.DisableCompilationReports()
    # Check that the tt_lib bindings take the correct path
    # to device epsilon constants
    assert tt_lib.device.EPS_GS == 0.001953125
    assert tt_lib.device.EPS_WHB0 == 1.1920899822825959e-07


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_device_arch():
    assert tt_lib.device.Arch.GRAYSKULL.name == "GRAYSKULL"
    assert tt_lib.device.Arch.WORMHOLE_B0.name == "WORMHOLE_B0"
    pass
