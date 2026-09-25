"""Failing to READ the baseline is not the same as there being no baseline.

The gate decides whether it has a reference by parsing a file, and swallows the failure:

    base = {}
    if base_path.exists():
        try:    base = json.loads(base_path.read_text())
        except: base = {}                  # <- indistinguishable from "no file"
    best = float(base.get("full_pipeline_ms", 0.0) or 0.0)

Downstream, `if best <= 0:` means "nothing to compare against", so the gate ESTABLISHES a new
baseline from whatever it just measured. Three different states funnel into that one branch:

    1. no file            -> establishing is correct
    2. file unparseable   -> establishing is WRONG; a reference exists and was lost
    3. ms missing or zero -> establishing is WRONG for the same reason

On gemma-3-12b-it the bar went from 34.9066 to 67.2294 (28.6 -> 14.9 tok/s/u) through that branch.
The gate log shows it plainly -- two rejections against a healthy bar, then a line with no best_ms
at all:

    diverged  61.0115  best_ms=34.9066
    diverged  69.7795  best_ms=34.9066
    ok        67.2294                     <- no best_ms: the reference was gone

The next run would then have graded every attempt against 14.9 tok/s/u and banked anything at all as
a ~20 ms win, written conclusive. It was stopped by hand.

Which of the three states occurred that night cannot now be proven -- the file has since been
rewritten. What CAN be proven from the code is that state 2 is reachable: `write_text` truncates
before it writes, so any reader inside that window sees a partial file, and stray perf_mcp processes
from another checkout were alive on this host.

So the fix does not depend on knowing which one happened. An existing-but-unreadable file must refuse
rather than re-baseline, and the write is made atomic so the window cannot exist. A silent
re-baseline becomes a visible failure.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _bar_path(mcp):
    return mcp._FULLPIPE_BASELINE_1CQ_PATH


# ---------------------------------------------------------------- the three states are distinguished


def test_a_corrupt_bar_is_reported_as_unreadable(mcp):
    """The state that cost 34.9066: the file is there, so a reference EXISTS."""
    _bar_path(mcp).write_text("{partial")
    best, mode, readable = mcp._read_fullpipe_bar()
    assert readable is False and best == 0.0


def test_an_absent_bar_is_readable_and_empty(mcp):
    """No file is the genuine no-baseline case, and establishing from it is correct."""
    best, mode, readable = mcp._read_fullpipe_bar()
    assert readable is True and best == 0.0


def test_a_healthy_bar_reads_back(mcp):
    _bar_path(mcp).write_text(json.dumps({"full_pipeline_ms": 34.9066, "mode": "trace+1cq"}))
    best, mode, readable = mcp._read_fullpipe_bar()
    assert readable is True and best == 34.9066 and mode == "trace+1cq"


def test_a_zeroed_bar_counts_as_unreadable(mcp):
    """A file whose ms is 0 or missing is a damaged reference, not an absent one -- same danger."""
    _bar_path(mcp).write_text(json.dumps({"mode": "trace+1cq"}))
    best, _mode, readable = mcp._read_fullpipe_bar()
    assert readable is False and best == 0.0


def test_an_empty_file_counts_as_unreadable(mcp):
    """Exactly what a reader sees mid-write_text: truncated to zero bytes."""
    _bar_path(mcp).write_text("")
    _best, _mode, readable = mcp._read_fullpipe_bar()
    assert readable is False


# ---------------------------------------------------------------- the write cannot be caught midway


def test_the_bar_is_written_atomically(mcp):
    """write_text truncates first, so a concurrent reader can see a partial file. A temp-then-rename
    means a reader sees either the old contents or the new, never half."""
    src = Path(mcp.__file__).read_text()
    i = src.index("def _write_fullpipe_bar")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "os.replace" in body, "the bar write is not atomic"


def test_the_atomic_write_round_trips(mcp):
    mcp._write_fullpipe_bar({"full_pipeline_ms": 33.5, "mode": "trace+1cq"})
    assert json.loads(_bar_path(mcp).read_text())["full_pipeline_ms"] == 33.5


def test_no_temp_file_is_left_behind(mcp):
    """A stray .tmp beside the bar would be picked up by nothing, but it signals a failed write."""
    mcp._write_fullpipe_bar({"full_pipeline_ms": 33.5, "mode": "trace+1cq"})
    leftovers = [p.name for p in _bar_path(mcp).parent.iterdir() if ".tmp" in p.name]
    assert leftovers == [], leftovers
