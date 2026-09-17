"""The report's ✓win column must require the win was SAVED, not just measured.

_record_committed_win (perf_mcp.git_commit) is the correct, single writer of a real win -- but the
report's actual win computation, summary._win_set -> measurements.winning_indices, never consulted
it. winning_indices' modern "stamped" path returns purely on fullpipe_delta_ms < 0, bypassing
is_win()'s own beat_baseline check entirely whenever any row in the list carries a stamped delta.
So an attempt row (which never carries a commit field at all -- that's a separate row
_record_committed_win writes) rendered "✓ win" the instant its measured delta looked good, with
zero regard for whether anything was ever saved.

Confirmed live on nvidia_nemotron_3_5_lightning_30b_a3b_bf16: three real edits (two matmul grid
configs, one BinaryNg fusion) rendered eight ✓win rows across their rungs, all with
beat_baseline=False and no commit anywhere in the run's actual git history -- the code was never
saved, only the measurement was recorded, and the report could not tell the difference.

The fix matches an attempt row back to a SEPARATE commit row by op_signature + kernel_kind (the
same two fields _op_match already keys every other op/rung comparison in this tool on), because
neither row alone carries both facts: the attempt has the delta, the commit row has the sha.

ABSENT ENTIRELY must mean unknown, not "not banked" -- commit_record is a newer field, and treating
its total absence as "nothing was ever saved" would blank every historical report to zero ticks,
including runs that really did commit real wins before this field existed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(_ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def perf_mcp():
    return _load("_report_ticks_perf_mcp", "cc_optimize/perf_mcp.py")


@pytest.fixture()
def summary(perf_mcp, monkeypatch):
    mod = _load("_report_ticks_summary", "cc_optimize/summary.py")
    monkeypatch.setattr(mod, "_perf_mcp", lambda: perf_mcp)
    return mod


def _attempt(sig, kind, delta_ms, **kw):
    r = {
        "op_signature": sig,
        "kernel_kind": kind,
        "fullpipe_delta_ms": delta_ms,
        "beat_baseline": False,
        "claimed_beat_baseline": True,
    }
    r.update(kw)
    return r


def _commit(sig, kind, **kw):
    r = {
        "op_signature": sig,
        "kernel_kind": kind,
        "beat_baseline": True,
        "commit_record": True,
        "commit": "sha1234",
        "kernel_detected_in_source": True,
    }
    r.update(kw)
    return r


# ---------------------------------------------------------------- the reported bug, reproduced


def test_a_banked_measurement_is_still_a_win(summary):
    op, kind = "MatmulDeviceOperation 32 x 2688 x 131072", "grid"
    attempts = [_attempt(op, kind, -109.32), _commit(op, kind)]
    assert summary._win_set(attempts) == {0}


def test_the_commit_row_itself_is_never_the_win(summary):
    """It carries no fullpipe_delta_ms, so it must never appear in the set on its own account --
    only the attempt row it validates may."""
    op, kind = "MatmulDeviceOperation 32 x 2688 x 131072", "grid"
    attempts = [_attempt(op, kind, -109.32), _commit(op, kind)]
    assert 1 not in summary._win_set(attempts)


def test_a_commit_on_a_different_rung_does_not_bank_this_one(summary):
    op = "MatmulDeviceOperation 32 x 2688 x 131072"
    attempts = [_attempt(op, "grid", -109.32), _commit(op, "dtype")]
    assert summary._win_set(attempts) == set()


def test_a_commit_on_a_different_op_does_not_bank_this_one(summary):
    attempts = [
        _attempt("MatmulDeviceOperation 32 x 2688 x 131072", "grid", -109.32),
        _commit("BinaryNgDeviceOperation", "grid"),
    ]
    assert summary._win_set(attempts) == set()


def test_two_real_wins_each_need_their_own_commit(summary):
    a_op, b_op = "MatmulDeviceOperation 32 x 2688 x 131072", "MatmulDeviceOperation 128 x 2688 x 1856"
    attempts = [
        _attempt(a_op, "grid", -109.32),
        _attempt(b_op, "grid", -83.5),
        _commit(a_op, "grid"),
        # b_op's commit is missing on purpose
    ]
    assert summary._win_set(attempts) == {0}


# ---------------------------------------------------------------- legacy ledgers are not blanked


def test_a_ledger_with_no_commit_rows_at_all_falls_back_unfiltered(summary):
    """A run recorded before commit_record existed must keep showing its real wins -- Voxtral: 434
    attempt rows, 30 genuinely committed wins, 0 rows carrying a commit marker."""
    attempts = [_attempt("Op A", "grid", -5.0), _attempt("Op B", "shard", -2.0)]
    assert summary._win_set(attempts) == {0, 1}


def test_mixed_ledger_only_filters_once_a_commit_marker_exists_anywhere(summary):
    """The moment even ONE commit_record row exists in the list, the run is known to be on the new
    system, and every win in it is held to the same standard -- not just the one with a matching
    commit."""
    op = "MatmulDeviceOperation 32 x 2688 x 131072"
    attempts = [
        _attempt(op, "grid", -109.32),
        _commit(op, "grid"),
        _attempt("Op B", "shard", -2.0),  # no commit anywhere for Op B/shard
    ]
    assert summary._win_set(attempts) == {0}


# ---------------------------------------------------------------- no perf_mcp reachable: fail open


def test_no_op_match_available_does_not_blank_the_report(summary, monkeypatch):
    monkeypatch.setattr(summary, "_perf_mcp", lambda: None)
    op = "MatmulDeviceOperation 32 x 2688 x 131072"
    attempts = [_attempt(op, "grid", -109.32), _commit(op, "grid")]
    assert summary._win_set(attempts) == {0}
