# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the prefill-matrix producer's measurement arithmetic
(models/demos/minimax_m3/scripts/prefill_matrix/matrix_producer.py): the steady-state window that excludes pipeline
fill/drain, and the per-request TTFT-under-load statistics. Synthetic chunk end times, no device."""

import importlib.util
import os

import pytest

_PKG = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "prefill_matrix")
os.environ.setdefault("PREFILL_MODEL", "minimax_m3")
_spec = importlib.util.spec_from_file_location("matrix_producer", os.path.join(_PKG, "matrix_producer.py"))
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def _ends(c0, n, period, t0=1000.0, fill_extra=0.0):
    """Chunk end times for chunks c0..c0+n-1 at a constant `period`; the first `stages` chunks may be slowed by
    `fill_extra` each to imitate the pipeline filling up."""
    ends, t = {}, t0
    for i in range(n):
        t += period + (fill_extra if i < 16 else 0.0)
        ends[c0 + i] = t
    return ends


def test_steady_state_constant_period_recovers_exact_throughput():
    chunk, stages, period = 5120, 16, 0.1
    ends = _ends(c0=100, n=240, period=period)
    s = mp.steady_state(ends, 100, 339, stages, real_per_chunk=640.0, chunk=chunk)
    assert s["mid_chunks"] == 240 - 2 * stages
    assert s["window_s"] == pytest.approx(s["mid_chunks"] * period)
    assert s["steady_processed_tps"] == pytest.approx(chunk / period)  # 51200 tok/s
    assert s["steady_new_tps"] == pytest.approx(640.0 / period)
    assert s["chunk_period_ms_median"] == pytest.approx(100.0)


def test_steady_state_excludes_slow_fill_chunks():
    """Slow chunks inside the first `stages` (fill) must not change the steady-state number."""
    stages, period = 16, 0.1
    clean = mp.steady_state(_ends(0, 240, period), 0, 239, stages, 5120.0, 5120)
    slow_fill = mp.steady_state(_ends(0, 240, period, fill_extra=0.5), 0, 239, stages, 5120.0, 5120)
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
    s = mp.steady_state(ends, c0, c0 + n - 1, stages, 1.0, 1)
    kept = [ends[c] - ends[c - 1] for c in range(c0 + stages, c0 + n - stages)]
    assert s["mid_chunks"] == len(kept) == n - 2 * stages
    assert s["window_s"] == pytest.approx(sum(kept))


def test_steady_state_too_short_returns_none():
    assert mp.steady_state(_ends(0, 30, 0.1), 0, 29, 16, 1.0, 5120) is None  # 30 - 32 < 4


def test_ttft_stats_and_completeness(expect_error):
    ends = _ends(0, 8, 0.1, t0=0.0)  # ends: 0.1, 0.2, ..., 0.8
    req = {
        (0, 0): {"c_first": 0, "c_last": 3, "t_push_first": 0.0},
        (1, 0): {"c_first": 4, "c_last": 7, "t_push_first": 0.05},
    }
    st = mp.ttft_stats(req, ends)
    assert st["ttft_under_load_ms_min"] == pytest.approx(400.0)
    assert st["ttft_under_load_ms_max"] == pytest.approx(750.0)
    assert st["ttft_under_load_ms_median"] == pytest.approx(575.0)
    del ends[7]
    with expect_error(AssertionError, "no completion row"):
        mp.ttft_stats(req, ends)
