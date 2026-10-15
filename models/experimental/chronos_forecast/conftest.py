# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for Chronos forecast tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def reset_seeds():
    """Reset RNG seeds for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
