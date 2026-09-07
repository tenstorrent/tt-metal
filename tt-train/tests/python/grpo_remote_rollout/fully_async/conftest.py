# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Local conftest for the fully_async tests.

Overrides the parent ``tests/python/grpo_remote_rollout/conftest.py``'s
``_set_fabric_2d`` fixture with a no-op. The threaded weight bridge +
rollout queue tests run WITHOUT fabric (matching the
``gsm8k_fully_async_training_example.py`` behavior). The mgd.textproto
under ``configurations/split_1_1/`` still declares a no-op fabric graph
so a future commit can flip fabric back on with no config churn.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _set_fabric_2d():
    """Override the parent's fixture to skip ttnn.set_fabric_config."""
    yield
