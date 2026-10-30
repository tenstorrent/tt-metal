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


# ------------------------------------------- the pair does not agree on the lever it names


def test_a_commit_naming_another_lever_still_banks_the_state_it_saved(summary):
    """op + rung is not a link the two rows can be held to, because only one of them chose it.

    The commit row takes its rung from whatever target is current when git_commit runs, not from the
    attempt that earned the win, so the pair routinely disagrees on BOTH fields -- measured on
    voxtral_4b_tts_2603, a fold win banked by a row calling itself structural-order and a kv-cache
    win banked by one calling itself trace-capture. Requiring agreement dropped five of seven real
    commits and rendered a 5.56x run as a single tick.

    What both rows do carry is the end-to-end reading at the moment of the commit, and the winning
    attempt is the one that produced it.
    """
    attempts = [
        _attempt("ReshapeViewDeviceOperation", "fold", -129.36, fullpipe_ms=242.2569),
        _commit("MatmulDeviceOperation 64 x 128 x 64", "structural-order", fullpipe_ms=242.2569),
    ]
    assert summary._win_set(attempts) == {0}


def test_a_win_no_commit_banked_is_still_not_a_win(summary):
    """The narrowing must keep doing its job: a measured improvement nothing saved never ticks."""
    attempts = [
        _attempt("MatmulDeviceOperation 2048 x 3072 x 9216", "dtype", -0.02, fullpipe_ms=137.5312),
        _commit("MatmulDeviceOperation 2048 x 3072 x 9216", "grid", fullpipe_ms=124.5104),
    ]
    assert summary._win_set(attempts) == set()


def test_each_commit_banks_its_own_win_and_not_a_neighbour(summary):
    """A run is a staircase of distinct readings, so the link must not smear across steps."""
    attempts = [
        _attempt("A", "fold", -129.36, fullpipe_ms=242.2569),
        _commit("X", "structural-order", fullpipe_ms=242.2569),
        _attempt("B", "fidelity", -52.91, fullpipe_ms=177.0374),
        _commit("Y", "grid", fullpipe_ms=177.0374),
        _attempt("C", "grid", -4.52, fullpipe_ms=172.5126),
    ]
    assert summary._win_set(attempts) == {0, 2}


def test_a_row_with_no_reading_is_not_banked_by_one_that_has_none_either(summary):
    """Two absent readings are not a match; that would bank every unmeasured attempt at once."""
    attempts = [_attempt("A", "fold", -12.0), _commit("X", "structural-order")]
    assert summary._win_set(attempts) == set()
