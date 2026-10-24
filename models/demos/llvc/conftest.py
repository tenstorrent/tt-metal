# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for LLVC TTNN tests."""

import pytest
import torch

import ttnn
from models.demos.llvc.tt.config import LLVC_L1_SMALL_SIZE, LLVC_TRACE_REGION_SIZE


@pytest.fixture(scope="session")
def torch_seed():
    torch.manual_seed(42)
    return 42


@pytest.fixture(scope="session")
def device():
    """Shared TTNN device for the test session."""
    # conv1d halo config tensors live in L1-small; reserve a trace region too.
    dev = ttnn.open_device(device_id=0, l1_small_size=LLVC_L1_SMALL_SIZE, trace_region_size=LLVC_TRACE_REGION_SIZE)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
