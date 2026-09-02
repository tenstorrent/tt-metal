# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the bounded-regression latency release gate."""

from __future__ import annotations

from models.autoports.meta_models_muse_glimmer_30b.doc.optimized_full_model.bench.latency_gate import evaluate


def payload(ttft: float, tpot: float, e2el: float) -> dict:
    return {
        "batch_size": 1,
        "osl": 512,
        "hf_advertised_context": 131072,
        "rows": [
            {
                "isl": 128,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "e2el_ms": e2el,
            }
        ],
    }


def test_equal_displayed_values_pass_on_first_sweep():
    report = evaluate(
        payload(64.957, 23.5941, 12121.553),
        [payload(64.96, 23.594, 12121.55)],
    )
    assert report["outcome"] == "pass"


def test_initial_regression_requires_two_retries():
    report = evaluate(payload(65.0, 23.59, 12121.6), [payload(65.2, 23.59, 12121.6)])
    assert report["outcome"] == "incomplete"
    assert report["rows"][0]["status"] == "retry_required"


def test_median_of_three_can_clear_an_initial_regression():
    baseline = payload(65.0, 23.59, 12121.6)
    candidates = [
        payload(65.2, 23.59, 12121.6),
        payload(64.8, 23.58, 12120.0),
        payload(64.9, 23.58, 12120.0),
    ]
    assert evaluate(baseline, candidates)["outcome"] == "pass"


def test_confirmed_median_regression_fails():
    baseline = payload(65.0, 23.59, 12121.6)
    candidates = [
        payload(65.1, 23.60, 12121.7),
        payload(65.2, 23.61, 12121.8),
        payload(65.3, 23.62, 12121.9),
    ]
    report = evaluate(baseline, candidates)
    assert report["outcome"] == "fail"
    assert report["rows"][0]["status"] == "fail"


def test_minor_regression_within_configured_percentage_passes():
    baseline = payload(100.0, 20.0, 1000.0)
    candidates = [
        payload(101.0, 20.2, 1010.0),
        payload(102.0, 20.4, 1020.0),
        payload(103.0, 20.6, 1030.0),
    ]
    report = evaluate(baseline, candidates, allowed_regression_percent=2.0)
    assert report["outcome"] == "pass"
    assert report["allowed_regression_percent"] == 2.0


def test_regression_beyond_configured_percentage_fails():
    baseline = payload(100.0, 20.0, 1000.0)
    candidates = [
        payload(102.1, 20.41, 1020.1),
        payload(102.2, 20.42, 1020.2),
        payload(102.3, 20.43, 1020.3),
    ]
    report = evaluate(baseline, candidates, allowed_regression_percent=2.0)
    assert report["outcome"] == "fail"


def test_small_absolute_ttft_regression_can_pass_percentage_limit():
    baseline = payload(65.0, 23.59, 12121.6)
    candidates = [
        payload(69.7, 23.60, 12128.1),
        payload(69.5, 23.60, 12128.1),
        payload(68.0, 23.58, 12119.4),
    ]
    report = evaluate(
        baseline,
        candidates,
        allowed_regression_percent=2.0,
        allowed_absolute_ms={"ttft_ms": 5.0},
    )
    assert report["outcome"] == "pass"
    assert report["rows"][0]["metrics"]["ttft_ms"]["absolute_regression_ms"] == 4.5


def test_negative_regression_allowance_is_rejected():
    try:
        evaluate(payload(100.0, 20.0, 1000.0), [payload(100.0, 20.0, 1000.0)], -0.1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative regression allowance should fail")

    try:
        evaluate(
            payload(100.0, 20.0, 1000.0),
            [payload(100.0, 20.0, 1000.0)],
            allowed_absolute_ms={"ttft_ms": -0.1},
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative absolute allowance should fail")
