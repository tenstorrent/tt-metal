"""Single source of truth for record timestamps.

All ledger / events / agent_call rows use utc_ts() so the format can never
drift across writers (the bug: before_loop wrote epoch floats, the loop wrote
local-time strings). One format, UTC, ISO-8601 with a trailing Z.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_ts() -> str:
    """Current time as a UTC ISO-8601 string, e.g. '2026-06-11T18:00:22Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
