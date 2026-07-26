"""Make the perf_automation package importable for tests without install.

Adds the perf_automation root (parent of this tests/ dir) to sys.path so
`import agent` resolves regardless of pytest's invocation cwd.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- hermetic by default -------------------------------------------------------------------------
# Several timers and classifiers now consult a Claude Code agent when they have no observation to
# scale from. That is correct in a run, but a TEST must never depend on an external process: it makes
# results non-deterministic (the agent's number varies), slow, and dependent on `claude` being on
# PATH. Tests that specifically exercise the agent path stub it and can opt in by unsetting this.

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_live_agent_calls(monkeypatch):
    monkeypatch.setenv("PERF_MCP_NO_AGENT_CLASSIFY", "1")
    yield
