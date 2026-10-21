# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for Timestral tests.

This directory and ``tests/`` deliberately have no ``__init__.py``: under the repo's
``--import-mode=importlib`` a package here would be registered as top-level ``timestral``
and shadow the upstream package the tests compare against.
"""

import numpy as np
import pytest
import torch

from models.experimental.timestral.common.timestral_src import require_timestral

require_timestral()


@pytest.fixture
def reset_seeds():
    """Reset RNG seeds for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
