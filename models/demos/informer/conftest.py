# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for Informer model tests."""

import pytest
import torch

import ttnn
from models.demos.utils.trace_region_sizes import build_trace_device_params


@pytest.fixture(scope="session")
def torch_seed():
    """Set torch seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture(scope="session")
def device():
    """Create one shared TTNN device for PCC test sessions."""
    dev = ttnn.open_device(device_id=0, **build_trace_device_params("informer"))
    dev.enable_program_cache()

    yield dev
    ttnn.close_device(dev)
