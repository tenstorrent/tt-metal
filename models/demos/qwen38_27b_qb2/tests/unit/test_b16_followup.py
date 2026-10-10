# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prevent benchmark-only or incomplete evidence from becoming serving claims."""

import copy

import pytest

from models.demos.qwen38_27b_qb2.demo.run_b16_followup import summarize_b16


def report():
    return dict(
        state="completed",
        cleanup_completed=True,
        replicas=1,
        output_tokens=128,
        cells=[
            dict(
                input_tokens=length,
                batch_per_replica=16,
                status="completed",
                samples=[
                    dict(
                        prefill_s=100.0,
                        decode_s=10.0,
                        elapsed_s=111.0,
                        ttft_s=[100.5] * 16,
                        output_sha256_per_replica=["a" * 64],
                    )
                    for _ in range(3)
                ],
            )
            for length in (32768, 16384)
        ],
    )


def test_all_in_and_decode_rates_are_different():
    rows = summarize_b16(report())
    assert rows[0]["decode_tsu"] == 12.7
    assert rows[0]["output_tps_including_prefill"] == 2048 / 111
    assert rows[0]["prefill_input_tps"] == 524288 / 100
    assert rows[0]["ttft_s"] == 100.5
    assert all(r["repeatable_tokens"] for r in rows)


@pytest.mark.parametrize("change", ["active", "cleanup", "batch", "missing", "nan", "zero", "ttft", "tokens"])
def test_rejects_incomplete_or_invalid_measurements(change, expect_error):
    data = copy.deepcopy(report())
    if change == "active":
        data["state"] = "running"
    elif change == "cleanup":
        data["cleanup_completed"] = False
    elif change == "batch":
        data["cells"][0]["batch_per_replica"] = 32
    elif change == "missing":
        data["cells"][0]["samples"].pop()
    elif change in ("nan", "zero"):
        data["cells"][0]["samples"][0]["decode_s"] = float("nan") if change == "nan" else 0
    elif change == "ttft":
        data["cells"][0]["samples"][0]["ttft_s"].pop()
    else:
        data["cells"][0]["samples"][0]["output_sha256_per_replica"] = []
    messages = {
        "active": "completed sweep",
        "cleanup": "completed sweep",
        "batch": "both B16 contexts",
        "missing": "three measured repetitions",
        "nan": "positive stage measurements",
        "zero": "positive stage measurements",
        "ttft": "per-user TTFT",
        "tokens": "output identity",
    }
    with expect_error(ValueError, messages[change]):
        summarize_b16(data)


def test_changed_output_is_visible_without_claiming_quality():
    data = report()
    data["cells"][0]["samples"][0]["output_sha256_per_replica"] = ["b" * 64]
    assert summarize_b16(data)[0]["repeatable_tokens"] is False
