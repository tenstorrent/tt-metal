# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE place the tool's durable temp state lives.

perf_automation keeps ~20 artifacts beside the run: baselines, gate verdicts, the knob cache, board
topology, throughput, profile caches, the measurement ledger. None is truncated on startup -- that is
deliberate, because a rerun must still see the ORIGINAL baseline -- which also means whatever else
writes there is what the next run reads.

Every one of those paths was built inline as ``Path(tempfile.gettempdir()) / "perf_mcp_....json"``
at 24 separate call sites. With no shared helper there was nowhere to redirect, so the test suite
wrote into the real state: a sentinel planted in perf_mcp_baseline_model_main.json was clobbered by
a plain `pytest tests/` with test data (wall_ms 20.15, device_ms 0.3651) -- a value shaped exactly
like the degenerate baselines that cost a morning to chase. Worse, several of those paths were
MODULE-LEVEL CONSTANTS, frozen at import before any fixture can run, so no test-side redirect could
reach them.

PERF_MCP_STATE_DIR redirects the whole namespace. Unset, this is exactly the previous behaviour, so
production is unchanged.

Resolved per call, never cached in a module constant: a constant is what made these unreachable.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def state_dir() -> Path:
    """The directory holding this tool's durable temp state."""
    return Path(os.environ.get("PERF_MCP_STATE_DIR") or tempfile.gettempdir())


def state_path(name: str) -> Path:
    """`name` inside the state directory."""
    return state_dir() / name
