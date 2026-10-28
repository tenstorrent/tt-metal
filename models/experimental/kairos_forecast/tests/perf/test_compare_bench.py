# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU unit checks for compare_bench.py (CONTRACT §4)."""

from models.experimental.kairos_forecast.tests.perf.compare_bench import compare


def _r(name="paper", prec="accuracy", ms=100.0, runs=(0.1, 0.1005), chip="c1", **kw):
    return {
        "name": name,
        "precision": prec,
        "layers": [2, 2],
        "device_ms": ms,
        "chip": chip,
        "split_runs": [{"device_s": s} for s in runs],
        **kw,
    }


def _cmp(cand, base, **kw):
    return compare(cand, base, kw.pop("target", "paper:accuracy"), "pcc_normalized", 0.995, 2.0, 3.0, False)[0]


def test_accept_and_small_gain():
    assert _cmp([_r(ms=95)], [_r()])
    assert not _cmp([_r(ms=99)], [_r()])  # 1 % < 2 %


def test_gain_inside_base_spread_rejects():
    assert not _cmp([_r(ms=97)], [_r(runs=(0.097, 0.1035))])  # 3 ms saved, 6.5 ms spread


def test_guard_regression_rejects():
    base = [_r(), _r("ragged", ms=50)]
    assert _cmp([_r(ms=95), _r("ragged", ms=51)], base)
    assert not _cmp([_r(ms=95), _r("ragged", ms=52)], base)  # 4 % slower


def test_cross_chip_and_pcc_and_missing_target():
    assert not _cmp([_r(ms=95, chip="c0")], [_r()])
    assert not _cmp([_r(ms=95, pcc_normalized=0.99)], [_r()])
    assert _cmp([_r(ms=95, pcc_normalized=0.999)], [_r()])
    assert not _cmp([_r("ragged", ms=10)], [_r("ragged")])
