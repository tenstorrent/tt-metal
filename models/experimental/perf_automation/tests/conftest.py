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


@_pytest.fixture(autouse=True)
def _private_temp_state(tmp_path_factory, monkeypatch):
    """No test may touch the tool's REAL temp state.

    The ledger was only one of these files. perf_automation keeps ~20 more durable artifacts beside
    it -- baselines, gate verdicts, the knob cache, board topology, throughput, profile caches -- and
    every one is built inline as ``Path(tempfile.gettempdir()) / "perf_mcp_...json"`` at 24 separate
    call sites. None is truncated on startup, so whatever a test leaves is what the next REAL run
    reads. Proven, not theorised: a sentinel planted in perf_mcp_baseline_model_main.json was
    clobbered by this suite with test data (wall_ms 20.15, device_ms 0.3651) -- a value shaped exactly
    like the degenerate baselines we spend runs chasing.

    Redirecting ``tempfile.tempdir`` covers all 24 sites at once, because every one of them resolves
    through gettempdir(), which returns this global when set. Per-site edits would touch production
    code paths (including a containment check that legitimately means the real temp dir) for no extra
    safety here.

    The factory's own basetemp is created BEFORE the patch, so pytest's tmp_path machinery is
    unaffected by the redirect.
    """
    import tempfile as _tempfile

    box = tmp_path_factory.mktemp("tmpstate")
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(box))
    # Belt and braces: anything still reaching gettempdir() directly (the containment check in
    # _reap_measurement_dir deliberately does) lands in the box too.
    monkeypatch.setattr(_tempfile, "tempdir", str(box))
    yield


@_pytest.fixture(autouse=True)
def _private_measurement_ledger(tmp_path_factory, monkeypatch):
    """Every test gets its OWN ledger.

    The ledger is deliberately durable and never truncated, so a test that writes to the real one
    leaves a PERMANENT anchor behind -- and the next test to render a report reads it. Not
    hypothetical: three `modeled_floor` rows leaked into the shared default file and made
    test_roofline_report report 341.47 ms whatever floor it was handed, but ONLY when run as part of
    the suite. An order-dependent failure in a report test is expensive to read, and a test that
    silently anchors a developer's real ledger is worse.

    Tests that exercise unkeyed/ambient behaviour delenv this themselves -- which is why the
    DIRECTORY is redirected too. PERF_MCP_LEDGER names ONE file, so deleting it sends any KEYED call
    in that same test straight back to the shared temp dir: test_floor_anchor_writeonce's deliberate
    delenv wrote a real perf_measurements_named_model_main.jsonl into /tmp beside a live run's
    ledger, where it was read as that run having split its anchors across two files. A delenv of
    PERF_MCP_LEDGER cannot switch PERF_MCP_LEDGER_DIR off.
    """
    box = tmp_path_factory.mktemp("ledger")
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_LEDGER", str(box / "measurements.jsonl"))
    yield
