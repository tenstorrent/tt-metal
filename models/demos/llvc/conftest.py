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
    # conv1d halo config tensors live in L1-small; reserve a trace region too.
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=23887872)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
