# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TTNN throughput helpers (alignment with ``tt-perf-report`` / ``perf*.txt`` stacks).

Stacked E2E summaries often show large DRAM-interleaved shares for:

- ``PermuteDeviceOperation`` (~26 %)
- ``ReshapeViewDeviceOperation`` (~22 %)

Both ``ttnn.reshape`` and ``ttnn.permute`` accept ``memory_config``; optionally placing outputs in
L1 can trim DRAM traffic. **Off by default** so demos keep DRAM-backed activations (quality /
L1 capacity). Perf tests enable these via ``perf/conftest.py``.

Environment:

- ``ACE_STEP_TM_OUTPUT_L1``: ``1`` enables L1 for **both** reshape and permute outputs (perf shortcut).
- ``ACE_STEP_RESHAPE_OUTPUT_L1``: ``1`` enables L1 only for ``ttnn.reshape``.
- ``ACE_STEP_PERMUTE_OUTPUT_L1``: ``1`` enables L1 only for ``ttnn.permute``.
"""

from __future__ import annotations

import os
from typing import Any


def _env_truthy(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw == "":
        return default
    if raw in ("0", "false", "no", "off", "n"):
        return False
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    return default


def _l1_enabled_for(name: str) -> bool:
    if _env_truthy("ACE_STEP_TM_OUTPUT_L1", False):
        return True
    return _env_truthy(name, False)


def _l1_memory_kwargs(ttnn: Any) -> dict:
    mc = getattr(ttnn, "L1_MEMORY_CONFIG", None)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_reshape_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.reshape`` to steer outputs toward L1 when enabled."""
    if not _l1_enabled_for("ACE_STEP_RESHAPE_OUTPUT_L1"):
        return {}
    return _l1_memory_kwargs(ttnn)


def ace_step_permute_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.permute`` to steer outputs toward L1 when enabled."""
    if not _l1_enabled_for("ACE_STEP_PERMUTE_OUTPUT_L1"):
        return {}
    return _l1_memory_kwargs(ttnn)
