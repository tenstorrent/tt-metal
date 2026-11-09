# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json

from gwdt_checks import Status
from gwdt_quad_verdict import load_hashes, load_ranks, overall, stage_verdicts

HOSTS = ["h0", "h1", "h2", "h3"]


def _rank(r, **kw):
    doc = {
        "rank": r,
        "host": HOSTS[r],
        "mode": "block_random",
        "fabric": "2d_torus_xy",
        "chunks_done": 20,
        "iterations": 10,
        "last_in": [9, 1],
        "last_out": [9, 1],
        "last_progress": 0.0,
        "first_bad": None,
        "mismatches": [],
        "error": None,
        "done": True,
    }
    return doc | kw


def _bad(what, devices, it=3, chunk=1):
    return {"iter": it, "chunk": chunk, "what": what, "devices": devices, "n_diff": 7}


def _ranks(over=None):
    over = over or {}
    return {r: over.get(r, _rank(r)) for r in range(4)}


def _status(rows):
    return [r.status for _, r in rows]


def test_all_pass():
    rows = stage_verdicts(_ranks(), {}, HOSTS, hung=False)
    assert _status(rows) == [Status.PASS] * 4
    assert [label for label, _ in rows] == ["stage 0 h0", "stage 1 h1", "stage 2 h2", "stage 3 h3"]
    assert overall(rows).status == Status.PASS


def test_output_mismatch_is_galaxy_compute():
    bad = _bad("output", [[1, 3]])
    rows = stage_verdicts(_ranks({2: _rank(2, first_bad=bad, mismatches=[bad])}), {}, HOSTS, hung=False)
    assert rows[2][1].status == Status.FAIL
    assert "galaxy 2 (h2) compute" in rows[2][1].reason
    assert "chips (1,3)" in rows[2][1].reason
    o = overall(rows)
    assert o.status == Status.FAIL and "galaxy 2 (h2) compute" in o.reason


def test_input_mismatch_is_link():
    bad = _bad("input", [[0, 0], [0, 1]])
    rows = stage_verdicts(_ranks({2: _rank(2, first_bad=bad, mismatches=[bad])}), {}, HOSTS, hung=False)
    assert rows[2][1].status == Status.FAIL
    assert "link 1→2" in rows[2][1].reason
    assert "chips (0,0),(0,1)" in rows[2][1].reason


def test_input_mismatch_after_upstream_output_mismatch_blames_upstream():
    out = _bad("output", [[1, 3]])
    inp = _bad("input", [[1, 3]])
    ranks = _ranks({1: _rank(1, first_bad=out, mismatches=[out]), 2: _rank(2, first_bad=inp, mismatches=[inp])})
    rows = stage_verdicts(ranks, {}, HOSTS, hung=False)
    assert "galaxy 1 (h1) compute" in overall(rows).reason
    assert "stage 1" in rows[2][1].reason


def test_stage0_input_is_h2d():
    bad = _bad("input", [[4, 2]])
    rows = stage_verdicts(_ranks({0: _rank(0, first_bad=bad, mismatches=[bad])}), {}, HOSTS, hung=False)
    assert "host→galaxy 0 (H2D)" in rows[0][1].reason


def test_cross_stage_hash_mismatch_is_link():
    dev = ["aa", "bb", "cc"]
    hashes = {
        0: [{"iter": 0, "chunk": 0, "in": ["t0", "t1", "t2"], "out": dev}],
        1: [{"iter": 0, "chunk": 0, "in": ["aa", "XX", "cc"], "out": dev}],
    }
    rows = stage_verdicts(_ranks(), hashes, HOSTS, hung=False)
    assert rows[1][1].status == Status.FAIL
    assert "link 0→1" in rows[1][1].reason
    # 3 devices only in this fixture; with TP=4 device 1 is (0,1).
    assert "chips (0,1)" in rows[1][1].reason
    assert rows[0][1].status == Status.PASS


def test_hang_link():
    ranks = _ranks(
        {
            0: _rank(0, done=False, last_in=[4, 0], last_out=[4, 0]),
            1: _rank(1, done=False, last_in=[4, 0], last_out=[4, 0]),
            2: _rank(2, done=False, last_in=[3, 1], last_out=[3, 1]),
            3: _rank(3, done=False, last_in=[3, 1], last_out=[3, 1]),
        }
    )
    rows = stage_verdicts(ranks, {}, HOSTS, hung=True)
    assert rows[2][1].status == Status.HANG
    assert "link 1→2" in rows[2][1].reason
    o = overall(rows)
    assert o.status == Status.HANG and "link 1→2" in o.reason


def test_hang_compute():
    ranks = _ranks({r: _rank(r, done=False) for r in range(4)} | {1: _rank(1, done=False, last_out=[9, 0])})
    rows = stage_verdicts(ranks, {}, HOSTS, hung=True)
    assert rows[1][1].status == Status.HANG
    assert "galaxy 1 (h1) compute" in rows[1][1].reason


def test_hang_with_no_stage_mid_chunk_is_still_hang():
    ranks = _ranks({r: _rank(r, done=False) for r in range(4)})
    rows = stage_verdicts(ranks, {}, HOSTS, hung=True)
    assert overall(rows, hung=True).status == Status.HANG


def test_hang_stage0_never_got_input():
    ranks = {r: _rank(r, done=False, last_in=None, last_out=None, chunks_done=0, iterations=0) for r in range(4)}
    rows = stage_verdicts(ranks, {}, HOSTS, hung=True)
    assert rows[0][1].status == Status.HANG
    assert "host→galaxy 0 (H2D)" in rows[0][1].reason


def test_missing_rank_fails():
    ranks = _ranks() | {3: None}
    rows = stage_verdicts(ranks, {}, HOSTS, hung=False)
    assert rows[3][1].status == Status.FAIL
    assert "rank 3 never started" in rows[3][1].reason
    assert rows[3][0] == "stage 3 h3"


def test_rank_error_fails():
    rows = stage_verdicts(_ranks({1: _rank(1, error="iteration 0 output is all zeros")}), {}, HOSTS, hung=False)
    assert rows[1][1].status == Status.FAIL
    assert "all zeros" in rows[1][1].reason


def test_not_done_without_hang_fails():
    rows = stage_verdicts(_ranks({3: _rank(3, done=False)}), {}, HOSTS, hung=False)
    assert rows[3][1].status == Status.FAIL
    assert "exited before the run finished" in rows[3][1].reason


def test_load_rank_tolerates_partial_json(tmp_path):
    (tmp_path / "rank0.json").write_text(json.dumps(_rank(0)))
    (tmp_path / "rank1.json").write_text('{"rank": 1, "host": "h1", "chu')
    ranks = load_ranks(tmp_path, 2, retries=2, delay=0)
    assert ranks[0]["host"] == "h0"
    assert ranks[1] is None


def test_load_hashes_skips_torn_last_line(tmp_path):
    good = {"iter": 0, "chunk": 0, "in": ["a"], "out": ["b"]}
    (tmp_path / "rank0.hashes.jsonl").write_text(json.dumps(good) + "\n" + '{"iter": 0, "ch')
    assert load_hashes(tmp_path, 1) == {0: [good]}
