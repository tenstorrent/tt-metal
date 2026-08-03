"""A measurement taken on a downclocked board is not a measurement of the code.

Every verdict this tool writes is a comparison: this candidate's milliseconds against a committed
best measured earlier. That comparison assumes one thing it never checked -- that both readings came
off a board running at the same speed.

On 2026-08-03 they did not. Runs 21 and 22 measured the gemma-3-12b-it baseline at 381.186 and
381.222 ms. Run 23 measured the SAME COMMIT at 632.331 ms. Nothing in the model changed; the board
had thermally throttled from 1350 MHz to 800 MHz mid-run:

    381.222 / 632.331 = 0.603        800 / 1350 = 0.593

Left running, every lever would have measured slow, lost to a bar set at full clock, and been written
to the ladder as a conclusive "no gain" -- and conclusive is forever, because the next run reads the
ladder and skips what is already settled. A throttled hour does not just waste an hour; it burns the
levers it touched.

This is the same defect class as the stale baseline: a real measurement compared against a reference
it does not belong to. The fix is the same shape -- refuse at the boundary rather than reason about
it afterwards. record_kernel_attempt already refuses an attempt owning no end-to-end measurement; it
now also refuses one taken below full clock.

Deliberately NOT a correction factor. Scaling 632 back by the clock ratio would invent a number no
board produced, and throttling is not uniform across an op mix. Refusing is honest; rescaling is the
fake win with extra arithmetic.

Fails OPEN. If tt-smi is missing, slow, or unparseable the attempt records normally -- an unreadable
clock is not evidence of a throttled one, and a telemetry hiccup must never be able to stall the loop.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.delenv("PERF_MCP_ALLOW_THROTTLED_ATTEMPT", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _clock(mcp, monkeypatch, mhz, limit=1350):
    """Pin the reported clock. None => telemetry unavailable."""
    monkeypatch.setattr(
        mcp,
        "_aiclk_health",
        lambda: None if mhz is None else {"min_mhz": mhz, "limit_mhz": limit, "ratio": mhz / limit},
    )


def _own_e2e(mcp, monkeypatch):
    """Satisfy the pre-existing end-to-end gate so this test isolates the clock one."""
    monkeypatch.setattr(
        mcp,
        "_attempt_fullpipe_verdict",
        lambda: {"own": True, "ms": 34.0, "ref": 34.99, "delta": -0.99, "win": True},
    )


# ---------------------------------------------------------------- the reported case


def test_a_throttled_attempt_is_refused(mcp, monkeypatch):
    """Run 23's board: 800 of 1350 MHz."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 800)
    r = mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 632.331, False, note="n")
    assert r.get("recorded") is False, r


def test_the_refusal_says_what_is_wrong(mcp, monkeypatch):
    """The agent has to know to WAIT, not to try a different lever."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 800)
    r = mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 632.331, False, note="n")
    msg = (r.get("refused") or "").lower()
    assert "800" in msg and "1350" in msg, msg


def test_a_full_clock_attempt_records(mcp, monkeypatch):
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 1350)
    assert mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 381.22, False, note="n").get(
        "recorded"
    )


def test_a_hair_under_the_ceiling_still_records(mcp, monkeypatch):
    """Boards jitter a few MHz around the ceiling. The guard is for a 40% collapse, not for noise --
    a tolerance too tight would refuse every attempt on a healthy board."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 1331)
    assert mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 381.9, False, note="n").get(
        "recorded"
    )


# ---------------------------------------------------------------- it fails open


def test_unreadable_telemetry_records_normally(mcp, monkeypatch):
    """An unreadable clock is not evidence of a throttled one, and tt-smi must never stall the loop."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, None)
    assert mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 381.22, False, note="n").get(
        "recorded"
    )


def test_the_helper_never_raises(mcp):
    """It shells out to tt-smi on a machine that may not have one."""
    out = mcp._aiclk_health()
    assert out is None or ("ratio" in out and "min_mhz" in out)


def test_the_override_lets_a_throttled_attempt_through(mcp, monkeypatch):
    """Mirrors PERF_MCP_ALLOW_UNMEASURED_ATTEMPT: an operator who knows why can still proceed."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 800)
    monkeypatch.setenv("PERF_MCP_ALLOW_THROTTLED_ATTEMPT", "1")
    assert mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 632.3, False, note="n").get(
        "recorded"
    )


# ---------------------------------------------------------------- a wedge still gets recorded


def test_a_wedged_attempt_is_still_recorded(mcp, monkeypatch):
    """A hang is a property of the candidate, not of the clock, and dropping the row hides the crash
    and invites the next run to re-derive it. Same carve-out the end-to-end gate makes."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 800)
    r = mcp.record_kernel_attempt(
        "MatmulDeviceOperation 32 x 3840 x 15360", "cpp", 0.0, False, note="wedged: kernel hung"
    )
    assert r.get("recorded"), r


# ---------------------------------------------------------------- provenance


def test_the_clock_is_stamped_on_a_recorded_attempt(mcp, monkeypatch):
    """So a later reader can tell a healthy measurement from one taken near the edge, instead of
    re-deriving thermal state from the numbers the way this bug had to be found."""
    _own_e2e(mcp, monkeypatch)
    _clock(mcp, monkeypatch, 1350)
    mcp.record_kernel_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 381.22, False, note="n")
    rows = mcp._load_attempts()
    assert rows and rows[-1].get("aiclk_mhz") == 1350, rows[-1] if rows else rows
