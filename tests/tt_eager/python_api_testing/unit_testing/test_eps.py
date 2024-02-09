# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest

import numpy as np
import tt_lib as ttl
from models.utility_functions import is_wormhole_b0


def test_run_sfpu_attr(device):
    assert ttl.device.EPS_GS == 0.001953125
    assert ttl.device.EPS_WHB0 == 1.1920899822825959e-07


def test_run_sfpu_eps(device):
    shape = [1, 1, 32, 32]
    value = [ttl.device.EPS_GS, ttl.device.EPS_WHB0][is_wormhole_b0()]
    assert np.isclose(value, device.sfpu_eps())


def test_run_sfpu_tensor(device):
    value = device.sfpu_eps()
    shape = [1, 1, 32, 32]
    eps = ttl.tensor.sfpu_eps(ttl.tensor.Shape(shape), ttl.tensor.Layout.ROW_MAJOR, device)
    eps = eps.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    passing = np.isclose(np.ones((1, 1, 32, 32)) * value, eps.float()).all()
    assert passing
