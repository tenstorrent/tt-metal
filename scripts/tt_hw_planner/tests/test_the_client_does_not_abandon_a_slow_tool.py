"""The MCP client must not give up on a tool call the server is still working on.

`claude -p` drops any single MCP tool call at its own built-in ceiling and reports nothing: the
server keeps running and its result is discarded, while the round watchdog stays quiet because the
round itself is healthy. perf_mcp's profile_model persists the roofline snapshot on its LAST
statement, so on an unoptimized model -- where one profile runs several times longer than on an
already-fast one -- that statement never ran, and RUN_REPORT silently fell back to a floor-only
roofline with no fidelity ladder and said nothing was wrong.

Both spawn paths must therefore hand the client an explicit ceiling: the bring-up loop here, and
the optimize engine's own `claude -p`, which does not go through this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.tt_hw_planner.cc_harness import _DEFAULT_AGENT_TIMEOUT_S, _mcp_tool_timeout_ms

_OVERRIDE = "TT_HW_PLANNER_CC_MCP_TOOL_TIMEOUT_MS"


def _without_override(monkeypatch):
    monkeypatch.delenv(_OVERRIDE, raising=False)


def test_a_ceiling_is_always_offered(monkeypatch):
    """No round budget must still yield a ceiling -- returning None is what reinstates the bug."""
    _without_override(monkeypatch)
    assert _mcp_tool_timeout_ms(None) == _DEFAULT_AGENT_TIMEOUT_S * 1000
    assert _mcp_tool_timeout_ms(0) == _DEFAULT_AGENT_TIMEOUT_S * 1000


def test_it_tracks_the_round_budget(monkeypatch):
    """Tied to the round so the two cannot contradict: a call cannot outlive the round that owns it."""
    _without_override(monkeypatch)
    assert _mcp_tool_timeout_ms(600) == 600_000


def test_an_unparseable_override_does_not_silently_restore_the_default(monkeypatch):
    """A typo must fall through to the derived value, not hand control back to the client."""
    monkeypatch.setenv(_OVERRIDE, "not-a-number")
    assert _mcp_tool_timeout_ms(600) == 600_000


def test_the_operator_can_still_hand_control_back(monkeypatch):
    """<=0 is the one documented way to reinstate the client's own default."""
    for raw in ("0", "-1"):
        monkeypatch.setenv(_OVERRIDE, raw)
        assert _mcp_tool_timeout_ms(600) is None


def test_an_explicit_override_wins(monkeypatch):
    monkeypatch.setenv(_OVERRIDE, "12345")
    assert _mcp_tool_timeout_ms(600) == 12345


def test_the_optimize_engine_sets_it_too(monkeypatch):
    """The optimize loop spawns its own `claude -p` and never calls run_cc_loop, so cc_env -- its
    single env owner -- must carry the ceiling as well. This is the path that actually broke."""
    import sys

    perf = str(Path(__file__).resolve().parents[3] / "models" / "experimental" / "perf_automation")
    if perf not in sys.path:
        sys.path.insert(0, perf)
    from cc_optimize.run import cc_env

    _without_override(monkeypatch)
    monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
    env = cc_env(Path(os.environ.get("TT_METAL_HOME") or "."), "0")
    assert int(env["MCP_TOOL_TIMEOUT"]) == _DEFAULT_AGENT_TIMEOUT_S * 1000


def test_an_operator_set_ceiling_is_never_clobbered(monkeypatch):
    import sys

    perf = str(Path(__file__).resolve().parents[3] / "models" / "experimental" / "perf_automation")
    if perf not in sys.path:
        sys.path.insert(0, perf)
    from cc_optimize.run import cc_env

    _without_override(monkeypatch)
    monkeypatch.setenv("MCP_TOOL_TIMEOUT", "999")
    assert cc_env(Path(os.environ.get("TT_METAL_HOME") or "."), "0")["MCP_TOOL_TIMEOUT"] == "999"
