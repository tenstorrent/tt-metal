# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time budgets for a model call that holds a device tool.

Both functions here were stranded when the Claude Agent SDK went: `device_call_timeout_s` lived in
`structural_agent.py` and `_operator_ceiling_s` in `sdk_retry.py`, and those files existed only to
drive SDK `query()` loops. The budgets themselves are not SDK-specific -- the CLI-driven perf-test
builder needs exactly the same numbers -- so they live here instead of dying with their old homes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def operator_ceiling_s() -> float:
    """The run-level timeout the operator configured, or 3 h."""
    mp = os.environ.get("PERF_MCP_MANIFEST")
    if mp:
        try:
            return float((json.loads(Path(mp).read_text()).get("config", {}) or {}).get("timeout") or 10800)
        except Exception:  # noqa: BLE001
            pass
    return 10800.0


def device_call_timeout_s() -> float:
    """Budget for one model call that holds a device tool.

    A fixed constant cannot adapt -- it is evaluated at import, before the manifest env is even set,
    and one hour is simultaneously far too long for a 3 s module and too short for a model whose
    device work legitimately runs longer. Same chain as the rest of the tool: this operation's
    observed p95, else an estimate from the model's evidence, else the operator's ceiling.
    """
    from .probes import adaptive_op_timeout

    return float(adaptive_op_timeout("agent", env_key="AGENT_DEVICE_CALL_TIMEOUT_S"))
