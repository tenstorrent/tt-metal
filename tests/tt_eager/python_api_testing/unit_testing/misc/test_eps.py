# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest

import numpy as np
import ttnn
from models.utility_functions import is_wormhole_b0
from ttnn.device import Arch


def test_run_sfpu_eps(device):
    shape = [1, 1, 32, 32]
    eps_mapping = {
        Arch.GRAYSKULL: 0.001953125,
        Arch.WORMHOLE_B0: 1.1920899822825959e-07,
        Arch.BLACKHOLE: 1.1920899822825959e-07,
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
