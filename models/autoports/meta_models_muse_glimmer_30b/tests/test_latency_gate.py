# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the strict latency release gate."""

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
