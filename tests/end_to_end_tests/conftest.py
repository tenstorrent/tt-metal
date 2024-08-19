# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
import numpy as np

import tt_lib


@pytest.fixture(scope="function")
def first_grayskull_device():
    device = tt_lib.device.CreateDevice(0)
    yield device

    tt_lib.device.CloseDevice(device)


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield
