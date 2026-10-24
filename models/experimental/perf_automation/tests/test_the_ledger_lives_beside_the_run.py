"""The keyed ledger defaults to the run's state dir, not to bare /tmp.

_ledger_dir resolved to `PERF_MCP_LEDGER_DIR or tempfile.gettempdir()`, and NOTHING in a real run
sets PERF_MCP_LEDGER_DIR -- the MCP config the optimize loop writes carries PERF_MCP_STATE_DIR only.
So every production run wrote and read its anchors in /tmp while every other artifact went to the
state dir:

    state dir   /home/ttuser/gemma3_perf_state/perf_measurements_gemma3_main.jsonl   <- the real one
    resolved    /tmp/perf_measurements_model_main.jsonl                              <- what it used

Two things follow, and the second is the one that showed up in every report.

anchor_value returned None on every real run, so the report's "THE LEDGER WINS" block -- which
exists specifically so the ceiling survives the optimize loop reverting the model directory -- could
never win. It fell through to the throughput snapshot every time. On gemma-3-12b-it that printed

    512 / 11.18 GB = 45.8 tok/s/u      the reverted-directory vintage
    512 / 12.00 GB = 42.7 tok/s/u      the operator-confirmed anchor

a 7% optimistic ceiling in all four reports written so far, and a utilization figure to match (67%
rather than 72%).

Note the filename in the resolved path: `model`, not `gemma3`. That is the separate
PERF_MCP_MODEL_NAME defaulting bug, and it means even the /tmp file was keyed wrong -- so the anchor
could not have been found there either.

PERF_MCP_LEDGER_DIR still redirects the whole namespace, which is what test isolation depends on
(redirecting one FILE is not enough: a test that delenvs it leaves keyed calls resolving back to the
shared temp dir and writing a real ledger beside a live run's). Only the DEFAULT changes, from
"somewhere in /tmp" to "beside the rest of this run's state".
"""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def meas(monkeypatch):
    monkeypatch.delenv("PERF_MCP_LEDGER_DIR", raising=False)
    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    import models.experimental.perf_automation.cc_optimize.measurements as m

    importlib.reload(m)
    return m


def test_it_defaults_to_the_state_dir(meas, monkeypatch, tmp_path):
    """The production case: the MCP config sets STATE_DIR and nothing else."""
    monkeypatch.delenv("PERF_MCP_LEDGER_DIR", raising=False)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    assert meas._ledger_dir() == tmp_path


def test_the_explicit_override_still_wins(meas, monkeypatch, tmp_path):
    """Test isolation depends on redirecting the whole namespace, so this must keep working."""
    other = tmp_path / "iso"
    other.mkdir()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(other))
    assert meas._ledger_dir() == other


def test_with_neither_set_it_falls_back_to_temp(meas, monkeypatch):
    """Unchanged behaviour when there is no run state to sit beside."""
    monkeypatch.delenv("PERF_MCP_LEDGER_DIR", raising=False)
    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    assert meas._ledger_dir() == Path(tempfile.gettempdir())


def test_the_anchor_is_findable_under_production_env(meas, monkeypatch, tmp_path):
    """The whole point: written and read in the same place, with only STATE_DIR set."""
    monkeypatch.delenv("PERF_MCP_LEDGER_DIR", raising=False)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    meas.anchor(meas.KIND_ACTIVE_BYTES, 12000.0, depth="token", model="gemma3", task="main")
    assert meas.anchor_value(meas.KIND_ACTIVE_BYTES, depth="token", model="gemma3", task="main") == 12000.0


def _production_env(monkeypatch, state):
    """The env a real run actually has: STATE_DIR only. conftest's autouse sandbox pins
    PERF_MCP_LEDGER_DIR (whole namespace) and PERF_MCP_LEDGER (one file) for every test, so both
    have to go before the production shape is reachable."""
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    monkeypatch.delenv("PERF_MCP_LEDGER_DIR", raising=False)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(state))


def test_the_ledger_file_sits_with_the_other_state(meas, monkeypatch, tmp_path):
    _production_env(monkeypatch, tmp_path)
    assert meas.ledger_path(model="gemma3", task="main").parent == tmp_path


def test_the_file_is_keyed_by_model_and_task(meas, monkeypatch, tmp_path):
    """perf_measurements_model_main.jsonl -- the unkeyed name -- is how one run's anchor was read as
    another's. The key must appear in the filename."""
    _production_env(monkeypatch, tmp_path)
    name = meas.ledger_path(model="gemma3", task="main").name
    assert "gemma3" in name and "main" in name, name
