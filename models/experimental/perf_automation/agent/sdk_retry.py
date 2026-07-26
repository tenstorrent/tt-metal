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
import os
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


# Hard per-call wall budget. The SDK spawns a CLI subprocess and streams from it; a stalled
# round (network / API / CLI) raises NOTHING, so asyncio.run(go()) would block FOREVER and
# freeze the whole loop (observed: a PLAN call hung 7+ min). asyncio.wait_for bounds each
# attempt; the TimeoutError is treated as transient and retried, so one stalled round can no
# longer freeze the loop. Generous default (a legit agentic PLAN reads many files over several
# turns) but finite; override via AGENT_CALL_TIMEOUT_S.
def _operator_ceiling_s() -> float:
    """The run-level timeout the operator configured, or 3 h."""
    import json as _json
    from pathlib import Path as _P

    mp = os.environ.get("PERF_MCP_MANIFEST")
    if mp:
        try:
            return float((_json.loads(_P(mp).read_text()).get("config", {}) or {}).get("timeout") or 10800)
        except Exception:  # noqa: BLE001
            pass
    return 10800.0


def _default_timeout() -> float:
    """Per-attempt agent-call budget, scaled from observed agent-call cost (BUG 4).
    A fixed 300 s governed EVERY SDK agent call regardless of model size."""
    try:
        from .probes import adaptive_op_timeout

        return float(adaptive_op_timeout("agent", env_key="AGENT_CALL_TIMEOUT_S"))
    except Exception:  # noqa: BLE001
        # Fixed 300 s was the defect (llama's agent calls run far longer). With no adaptive path
        # available, concede the operator ceiling rather than invent a tight number.
        return float(os.environ.get("AGENT_CALL_TIMEOUT_S", "") or _operator_ceiling_s())


def is_transient(exc) -> bool:
    # a bounded-wait timeout is, by definition, a recoverable stall -> retry
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    s = str(exc).lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


def run_with_retry(go, reset=None, attempts=3, base_sleep=0.5, timeout=None):
    """Run async query loop `go` (a no-arg coroutine function) with retries on transient SDK
    transport errors AND a hard per-attempt `timeout` (seconds; default AGENT_CALL_TIMEOUT_S or
    300). `reset()` clears accumulators before each attempt. A stalled round is cancelled at
    `timeout` and retried (the cancel propagates into the SDK async generator, which terminates
    its CLI subprocess). Re-raises immediately on a non-transient error, or after `attempts` —
    callers degrade gracefully (PLAN improvises, SELECT falls back to untried[0]) rather than
    hanging forever.
    """
    if timeout is None:
        timeout = _default_timeout()
    last = None
    for i in range(attempts):
        if reset is not None:
            reset()
        try:
            asyncio.run(asyncio.wait_for(go(), timeout))
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1 and is_transient(exc):
                time.sleep(base_sleep * (i + 1))
                continue
            raise
    if last is not None:
        raise last
