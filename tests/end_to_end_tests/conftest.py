# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
import numpy as np

import ttnn.deprecated


@pytest.fixture(scope="function")
def first_grayskull_device():
    device = ttnn.deprecated.device.CreateDevice(0)
    yield device

    ttnn.deprecated.device.CloseDevice(device)


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield
