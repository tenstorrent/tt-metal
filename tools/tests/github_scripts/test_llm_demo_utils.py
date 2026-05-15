# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.utils.llm_demo_utils import verify_perf


def test_verify_perf_uses_prefill_tolerance_for_ttft_when_decode_tolerance_present():
    expected_perf_metrics = {
        "prefill_time_to_first_token": 100.0,
        "prefill_tolerance": 1.25,
        "decode_t/s/u": 100.0,
        "decode_tolerance": 1.05,
    }
    measurements = {
        "prefill_time_to_first_token": 0.082,
        "decode_t/s/u": 104.0,
    }

    verify_perf(
        measurements=measurements,
        expected_perf_metrics=expected_perf_metrics,
        high_tol_percentage=1.05,
        expected_measurements={
            "prefill_time_to_first_token": True,
            "decode_t/s/u": True,
        },
    )


def test_verify_perf_keeps_ttft_seconds_targets_back_compatible():
    expected_perf_metrics = {
        "prefill_time_to_first_token": 0.12,
        "prefill_tolerance": 1.25,
    }
    measurements = {
        "prefill_time_to_first_token": 0.11,
    }

    verify_perf(
        measurements=measurements,
        expected_perf_metrics=expected_perf_metrics,
        expected_measurements={
            "prefill_time_to_first_token": True,
        },
    )


def test_verify_perf_supports_legacy_ttft_metric_name():
    expected_perf_metrics = {
        "prefill_time_to_token": 0.12,
        "prefill_tolerance": 1.25,
    }
    measurements = {
        "prefill_time_to_token": 0.11,
    }

    verify_perf(
        measurements=measurements,
        expected_perf_metrics=expected_perf_metrics,
        expected_measurements={
            "prefill_time_to_token": True,
        },
    )


def test_verify_perf_rejects_legacy_and_canonical_ttft_in_same_dict():
    with pytest.raises(ValueError, match="legacy and canonical forms"):
        verify_perf(
            measurements={
                "prefill_time_to_token": 0.11,
                "prefill_time_to_first_token": 0.11,
            },
            expected_perf_metrics={"prefill_time_to_first_token": 0.12},
            expected_measurements={"prefill_time_to_first_token": True},
        )
