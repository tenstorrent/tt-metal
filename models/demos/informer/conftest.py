# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for Informer model tests."""

import pytest
import torch

import ttnn


@pytest.fixture(scope="session")
def torch_seed():
    """Set torch seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture(scope="session")
def device():
    """Create one shared TTNN device for PCC test sessions."""
    dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20)
    dev.enable_program_cache()

    yield dev
    ttnn.close_device(dev)
