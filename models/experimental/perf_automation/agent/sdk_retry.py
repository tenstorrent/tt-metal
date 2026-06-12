"""Retry wrapper for the Claude Agent SDK query loop.

The SDK spawns a CLI subprocess; transient transport failures surface as e.g.
"Fatal error in message reader: Command failed with exit code 129" (SIGHUP) or
CLIConnection / Process errors. A single blip would otherwise kill an agentic
stage (PLAN silently degrading to improvise, SELECT losing its choice, etc.).
This retries the whole query loop a few times on transient errors, resetting
the output accumulators before each attempt.
"""

from __future__ import annotations

import asyncio
import time

_TRANSIENT_MARKERS = (
    "exit code 129",
    "exit code 143",
    "message reader",
    "cliconnection",
    "process error",
    "processerror",
    "transport",
    "broken pipe",
)


def is_transient(exc) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


def run_with_retry(go, reset=None, attempts=3, base_sleep=0.5):
    """Run async query loop `go` (a no-arg coroutine function) with retries on
    transient SDK transport errors. `reset()` clears accumulators before each
    attempt. Re-raises immediately on a non-transient error, or after `attempts`.
    """
    last = None
    for i in range(attempts):
        if reset is not None:
            reset()
        try:
            asyncio.run(go())
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1 and is_transient(exc):
                time.sleep(base_sleep * (i + 1))
                continue
            raise
    if last is not None:
        raise last
