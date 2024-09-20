# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest

import numpy as np
import ttnn
from models.utility_functions import is_wormhole_b0
from ttnn.device import Arch


def test_run_sfpu_attr(device):
    assert ttnn.device.EPS_GS == 0.001953125
    assert ttnn.device.EPS_WHB0 == 1.1920899822825959e-07
    assert ttnn.device.EPS_BH == 1.1920899822825959e-07


def test_run_sfpu_eps(device):
    shape = [1, 1, 32, 32]
    eps_mapping = {
        Arch.GRAYSKULL: ttnn.device.EPS_GS,
        Arch.WORMHOLE_B0: ttnn.device.EPS_WHB0,
        Arch.BLACKHOLE: ttnn.device.EPS_BH,
    }
    value = eps_mapping[device.arch()]
    assert np.isclose(value, device.sfpu_eps())


def test_run_sfpu_tensor(device):
    value = device.sfpu_eps()
    shape = [1, 1, 32, 32]
    eps = ttnn.full(ttnn.Shape(shape), value)
    eps = eps.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    passing = np.isclose(np.ones((1, 1, 32, 32)) * value, eps.float()).all()
    assert passing
