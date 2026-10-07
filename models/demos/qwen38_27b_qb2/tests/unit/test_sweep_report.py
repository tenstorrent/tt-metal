# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measurement accounting must not turn compilation or projections into results."""

from models.demos.qwen38_27b_qb2.tests.sweep_report import MAX_CONTEXT, MAX_POOL_TOKENS, make_plan, summarize


def test_plan_keeps_output_room_and_distinguishes_untested_capacity():
    plan = make_plan()
    assert sum(row["status"] == "queued" for row in plan["cells"]) == 27
    assert not any("summary" in row for row in plan["cells"])
    for row in plan["cells"]:
        assert row["input_tokens"] + plan["output_tokens"] - 1 <= MAX_CONTEXT
        assert (row["status"] == "queued") == (row["pool_tokens_per_replica"] <= MAX_POOL_TOKENS)
    assert {row["status"] for row in plan["cells"]} == {"queued", "capacity_guard"}


def test_galaxy_plan_requires_actual_eight_replica_measurements():
    one, eight = make_plan(), make_plan(8)
    assert eight["chips"] == 32
    for original, galaxy in zip(one["cells"], eight["cells"]):
        assert galaxy["concurrency"] == original["concurrency"] * 8
        assert galaxy["pool_tokens_per_replica"] == original["pool_tokens_per_replica"]
        assert "summary" not in galaxy


def test_metrics_keep_decode_and_end_to_end_time_separate():
    samples = [dict(decode_s=2, elapsed_s=4, ttft_s=[1, 2], trace_captures=0) for _ in range(3)]
    metrics = summarize(samples, concurrency=2, output_tokens=101)
    assert metrics["tokens_per_second_per_user"] == 50
    assert metrics["aggregate_decode_tokens_per_second"] == 100
    assert metrics["aggregate_e2e_tokens_per_second"] == 50.5
    assert metrics["tpot_ms"] == 20
    assert metrics["ttft_p50_s"] == 1.5 and metrics["ttft_p90_s"] == 2


def test_cold_trace_capture_cannot_be_reported_as_warm(expect_error):
    sample = dict(decode_s=2, elapsed_s=4, ttft_s=[1], trace_captures=2)
    with expect_error(ValueError, "warmup is incomplete"):
        summarize([sample], concurrency=1, output_tokens=128)


def test_missing_request_latencies_rejected(expect_error):
    sample = dict(decode_s=2, elapsed_s=4, ttft_s=[1], trace_captures=0)
    with expect_error(ValueError, "Invalid measurement"):
        summarize([sample], concurrency=2, output_tokens=128)
