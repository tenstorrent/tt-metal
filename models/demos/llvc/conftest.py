# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for LLVC TTNN tests."""

import pytest
import torch

import ttnn


@pytest.fixture(scope="session")
def torch_seed():
    torch.manual_seed(42)
    return 42


@pytest.fixture(scope="session")
def device():
    """Shared TTNN device for the test session."""
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
