# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the prefill-matrix producer's measurement arithmetic
(models/demos/minimax_m3/scripts/prefill_matrix/matrix_math.py, stdlib only): the steady-state window that excludes
pipeline fill/drain, the per-request TTFT-under-load statistics and the timing-CSV reader. No ttnn, no device."""

import importlib.util
import os

import pytest

_PKG = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "prefill_matrix")
_spec = importlib.util.spec_from_file_location("matrix_math", os.path.join(_PKG, "matrix_math.py"))
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def _ends(c0, n, period, t0=1000.0, fill_extra=0.0, stages=16):
    """Chunk end times for chunks c0..c0+n-1 at a constant `period`; the first `stages` chunks may be slowed by
    `fill_extra` each to imitate the pipeline filling up."""
    ends, t = {}, t0
    for i in range(n):
        t += period + (fill_extra if i < stages else 0.0)
        ends[c0 + i] = t
    return ends


def _tokens(c0, n, per_chunk):
    return {c0 + i: per_chunk for i in range(n)}


def test_steady_state_constant_period_recovers_exact_throughput():
    chunk, stages, period = 5120, 16, 0.1
    ends = _ends(c0=100, n=240, period=period)
    s = mp.steady_state(ends, 100, 339, stages, _tokens(100, 240, 640), chunk)
    assert s["mid_chunks"] == 240 - 2 * stages
    assert s["window_s"] == pytest.approx(s["mid_chunks"] * period)
    assert s["steady_processed_tps"] == pytest.approx(chunk / period)  # 51200 tok/s
    assert s["steady_new_tps"] == pytest.approx(640.0 / period)
    assert s["chunk_period_ms_median"] == pytest.approx(100.0)


def test_steady_state_excludes_slow_fill_chunks():
    """Slow chunks inside the first `stages` (fill) must not change the steady-state number."""
    stages, period = 16, 0.1
    tok = _tokens(0, 240, 5120)
    clean = mp.steady_state(_ends(0, 240, period), 0, 239, stages, tok, 5120)
    slow_fill = mp.steady_state(_ends(0, 240, period, fill_extra=0.5, stages=stages), 0, 239, stages, tok, 5120)
    assert slow_fill["steady_processed_tps"] == pytest.approx(clean["steady_processed_tps"])
    assert slow_fill["mid_chunks"] == clean["mid_chunks"] == 208


def test_steady_state_window_boundaries_are_off_by_none():
    """The window starts at the end of the chunk BEFORE the first kept chunk, so it spans exactly mid periods:
    make the periods distinguishable and check the window sums exactly the kept periods."""
    c0, n, stages = 10, 40, 4
    ends, t = {}, 0.0
    for i in range(n):
        t += 1.0 + i * 0.01  # strictly increasing periods
        ends[c0 + i] = t
    s = mp.steady_state(ends, c0, c0 + n - 1, stages, _tokens(c0, n, 1), 1)
    kept = [ends[c] - ends[c - 1] for c in range(c0 + stages, c0 + n - stages)]
    assert s["mid_chunks"] == len(kept) == n - 2 * stages
    assert s["window_s"] == pytest.approx(sum(kept))


def test_steady_state_too_short_returns_none():
    assert mp.steady_state(_ends(0, 30, 0.1), 0, 29, 16, _tokens(0, 30, 1), 5120) is None  # 30 - 32 < 4
    assert mp.steady_state(_ends(0, 35, 0.1), 0, 34, 16, _tokens(0, 35, 1), 5120) is None  # 35 - 32 = 3 < 4
    assert mp.steady_state(_ends(0, 36, 0.1), 0, 35, 16, _tokens(0, 36, 1), 5120)["mid_chunks"] == 4


def test_steady_state_ragged_chunks_count_actual_new_tokens():
    """N=6900 = 5120 + 1780 per request: the kept window must sum the real new tokens of the kept chunks, not a
    per-chunk average (the window is not a whole number of requests)."""
    chunk, stages, period, n = 5120, 2, 0.1, 21
    ends = _ends(0, n, period)
    tok = {c: (chunk if c % 2 == 0 else 1780) for c in range(n)}  # request = chunks (2i, 2i+1); 21st chunk is full
    s = mp.steady_state(ends, 0, n - 1, stages, tok, chunk)
    kept = range(stages, n - stages)  # 2..18 -> 9 full + 8 ragged
    assert s["mid_chunks"] == len(kept) == 17
    assert s["steady_new_tps"] == pytest.approx(sum(tok[c] for c in kept) / (17 * period))
    assert s["steady_new_tps"] != pytest.approx(17 * (6900 / 2) / (17 * period))  # the average would be wrong here


def test_ttft_stats_and_completeness(expect_error):
    ends = _ends(0, 8, 0.1, t0=0.0)  # ends: 0.1, 0.2, ..., 0.8
    req = {
        (0, 0): {"c_first": 0, "c_last": 3, "t_push_first": 0.0},
        (1, 0): {"c_first": 4, "c_last": 7, "t_push_first": 0.05},
    }
    st = mp.ttft_stats(req, ends)
    assert st["ttft_under_load_requests"] == 2
    assert st["ttft_under_load_ms_min"] == pytest.approx(400.0)
    assert st["ttft_under_load_ms_max"] == pytest.approx(750.0)
    assert st["ttft_under_load_ms_median"] == pytest.approx(575.0)
    assert st["ttft_under_load_ms_p90"] == pytest.approx(750.0)  # nearest rank of 2 -> the larger one
    # requests whose first chunk entered a filling pipeline (c_first < c_full) are excluded
    st = mp.ttft_stats(req, ends, c_full=4)
    assert st["ttft_under_load_requests"] == 1
    assert st["ttft_under_load_ms_median"] == pytest.approx(750.0)
    # ... unless that would leave nothing: then all requests are used
    assert mp.ttft_stats(req, ends, c_full=100)["ttft_under_load_requests"] == 2
    del ends[7]
    with expect_error(AssertionError, "no completion row"):
        mp.ttft_stats(req, ends)


def test_ttft_stats_p90_is_nearest_rank():
    ends = {i: float(i + 1) for i in range(10)}  # request i ends at i+1 s, pushed at 0 -> TTFT (i+1)*1000 ms
    req = {(0, i): {"c_first": i, "c_last": i, "t_push_first": 0.0} for i in range(10)}
    st = mp.ttft_stats(req, ends)
    assert st["ttft_under_load_ms_p90"] == pytest.approx(9000.0)  # ceil(0.9*10) = 9th of 10 sorted values
    assert st["ttft_under_load_ms_median"] == pytest.approx(5500.0)


def test_read_rank_csv_accepts_only_complete_lines(tmp_path):
    assert mp.read_rank_csv(str(tmp_path), 15) == {}  # no file yet
    f = tmp_path / "rank15.csv"
    f.write_text("15,0,1000.0,50.0\n15,1,1000.1\nheader,junk\n15,2,1000.2,60.0\n15,3,1000.3,61")
    rows = mp.read_rank_csv(str(tmp_path), 15)
    assert rows == {0: (1000.0, 50.0), 2: (1000.2, 60.0)}  # short/junk rows skipped; torn last line ignored
    f.write_text(f.read_text() + ".5\n")
    assert mp.read_rank_csv(str(tmp_path), 15)[3] == (1000.3, 61.5)
