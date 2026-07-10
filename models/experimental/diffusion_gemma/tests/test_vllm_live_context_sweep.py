# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.experimental.diffusion_gemma.doc.vllm_integration import live_context_sweep as sweep


def test_server_launch_forces_optimized_trace_stack_and_full_prefill_budget():
    args = SimpleNamespace(
        tt_metal=Path("/tt-metal"),
        checkpoint=Path("/checkpoint"),
        mesh="P150x4",
        trace_region_size=sweep.DEFAULT_TRACE_REGION_SIZE,
        max_model_len=4096,
        max_denoise_steps=12,
        host="127.0.0.1",
        port=8000,
    )

    _, selected = sweep._server_env(args)
    assert selected["DG_VLLM_TRACE"] == "1"
    assert selected["DG_VLLM_GUMBEL_MODE"] == "argmax"
    assert selected["DG_SPARSE_MOE"] == "1"
    assert selected["DG_DEDUP_ARGMAX"] == "1"
    assert selected["DG_SPARSE_MOE_TUNED"] == "1"
    assert selected["DG_VLLM_MAX_DENOISE_STEPS"] == "12"

    command = sweep._server_command(args)
    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"


def test_request_summary_proves_capture_replay_and_release_contract():
    max_denoise_steps = 12
    trace_ids = [str(i) for i in range(max_denoise_steps)]
    model_events = [
        {
            "event": "session_create",
            "denoise_path": "traced_denoise_block",
            "max_denoise_steps": max_denoise_steps,
        },
        {
            "event": "prefill_block0",
            "prompt_len": 32,
            "cache_len": 32,
            "prefill_s": 1.0,
            "ttft_s": 10.0,
            "block_latency_s": 9.0,
            "denoise_latency_s": 8.0,
            "commit_latency_s": 1.0,
            "denoise_steps": max_denoise_steps,
            "committed_tokens": 256,
            "start_pos": 32,
            "next_pos": 288,
            "dram": {"used_gib": 14.0, "free_gib": 7.0},
        },
    ]
    for block_idx in range(1, 4):
        model_events.append(
            {
                "event": "decode_block",
                "block_idx": block_idx,
                "block_latency_s": 12.0 + block_idx,
                "denoise_latency_s": 11.0 + block_idx,
                "commit_latency_s": 1.0,
                "denoise_steps": max_denoise_steps,
                "committed_tokens": 256,
                "start_pos": 32 + block_idx * 256,
                "next_pos": 32 + (block_idx + 1) * 256,
            }
        )
    model_events.append(
        {
            "event": "request_release",
            "dram": {"used_gib": 13.5, "free_gib": 7.5},
        }
    )

    trace_events = [
        {
            "event": "capture",
            "capture_events": 1,
            "traces_captured": max_denoise_steps,
            "trace_ids": trace_ids,
        }
    ]
    for block_idx in range(4):
        trace_events.append(
            {
                "event": "replay",
                "capture_events": 1,
                "traces_captured": max_denoise_steps,
                "trace_ids": trace_ids,
                "captured_this_block": block_idx == 0,
            }
        )
    trace_events.append(
        {
            "event": "release",
            "capture_events": 1,
            "replay_blocks": 4,
            "execute_trace_calls": 4 * max_denoise_steps,
        }
    )
    response = {
        "id": "cmpl-test",
        "usage": {"prompt_tokens": 32, "completion_tokens": 1024, "total_tokens": 1056},
        "choices": [{"finish_reason": "length", "text": "", "token_ids": [1] * 1024}],
    }

    result = sweep._request_summary(
        target=32,
        prompt_meta={"prompt_token_sha256": "abc"},
        response=response,
        request_wall_s=50.0,
        model_events=model_events,
        trace_events=trace_events,
        blocks_requested=4,
        max_denoise_steps=max_denoise_steps,
        log_segment="traced serving decode enabled",
        repeat_index=0,
    )

    assert result["position_progression"] == [32, 288, 544, 800, 1056]
    assert result["trace"]["metal_traces_captured"] == max_denoise_steps
    assert result["trace"]["steady_execute_trace_calls"] == 3 * max_denoise_steps
    assert result["trace"]["released"] is True
    assert result["steady"]["mean_s"] == 14.0
    assert result["steady"]["p99_s"] == 15.0
    assert result["steady"]["blocks_per_s"] == round(1 / 14, 6)
    assert result["steady"]["commit_mean_s"] == 1.0
    assert result["steady"]["output_tokens_per_s"] == round(256 / 14, 6)

    aggregate = sweep._aggregate_target_requests(32, [result, result])
    assert aggregate["measured_requests"] == 2
    assert aggregate["steady_block_samples"] == 6
    assert aggregate["p99_s"] == 15.0
    assert aggregate["commit_p99_s"] == 1.0


@pytest.mark.parametrize("value", ["0", "49", "not-an-int"])
def test_live_sweep_rejects_invalid_step_budget(value, expect_error):
    with expect_error(argparse.ArgumentTypeError):
        sweep._parse_max_denoise_steps(value)


def test_vllm_adapter_applies_validated_step_override(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.tt.generator_vllm import _with_vllm_max_denoise_steps

    monkeypatch.setenv("DG_VLLM_MAX_DENOISE_STEPS", "20")
    assert _with_vllm_max_denoise_steps(DiffusionConfig()).max_denoise_steps == 20

    for invalid in ("0", "49", "not-an-int"):
        monkeypatch.setenv("DG_VLLM_MAX_DENOISE_STEPS", invalid)
        with expect_error(ValueError, match=r"\[1, 48\]"):
            _with_vllm_max_denoise_steps(DiffusionConfig())


def test_compact_step_sweep_evidence_matches_passed_raw_requests():
    evidence_dir = Path(__file__).parents[1] / "doc" / "vllm_integration"
    summary = json.loads((evidence_dir / "live_denoise_step_sweep_results_20260710.json").read_text())

    expected_steps = [1, 4, 8, 12, 16, 20, 24, 32, 40, 48]
    assert summary["status"] == "passed"
    assert [row["max_denoise_steps"] for row in summary["rows"]] == expected_steps

    for row in summary["rows"]:
        raw = json.loads((evidence_dir / row["raw_json"]).read_text())
        request = raw["requests"][0]
        aggregate = raw["aggregates"][0]
        steps = row["max_denoise_steps"]

        assert raw["status"] == "passed"
        assert raw["max_denoise_steps"] == steps
        assert raw["server"]["env"]["DG_VLLM_MAX_DENOISE_STEPS"] == str(steps)
        assert raw["server"]["log_sha256"] == row["server_log_sha256"]
        assert request["actual_logical_prompt_tokens"] == 256
        assert request["committed_tokens"] == 1024
        assert request["compile_markers_in_request"] == 0
        assert request["denoise_steps_per_block"] == [steps] * 4
        assert row["model_build_s"] == raw["server"]["model_build"]["model_build_s"]
        assert row["prefill_s"] == request["prefill_s"]
        assert row["capture_inclusive_block0_ttft_s"] == request["block0_ttft_s"]
        assert row["block_latencies_s"] == [block["latency_s"] for block in request["blocks"]]
        assert row["steady"]["mean_s"] == aggregate["mean_s"]
        assert row["steady"]["output_tokens_per_s"] == aggregate["output_tokens_per_s"]
        assert row["steady"]["denoise_ms_per_step"] == aggregate["denoise_ms_per_step"]
        assert row["trace"]["distinct_traces_captured"] == steps
        assert row["trace"]["execute_trace_calls_total"] == 4 * steps
        assert request["trace"]["capture_events"] == 1
        assert request["trace"]["block_replays_total"] == 4
        assert request["trace"]["steady_block_replays"] == 3
        assert request["trace"]["released"] is True
        assert request["trace"]["eager_fallback"] is False
        assert request["trace"]["recapture_after_block0"] is False

    excluded = summary["excluded_runs"]
    assert len(excluded) == 1
    failed = json.loads((evidence_dir / excluded[0]["raw_json"]).read_text())
    assert failed["status"] == "failed"
    assert failed["requests"] == []
    assert excluded[0]["included_in_metrics"] is False


def test_warmed_context_handoff_is_finalized_with_completed_targets():
    evidence_dir = Path(__file__).parents[1] / "doc" / "vllm_integration"
    raw = json.loads((evidence_dir / "live_context_sweep_msl4096_warmed.json").read_text())

    assert raw["status"] == "interrupted"
    assert raw["interruption"]["completed_targets"] == [32, 256, 1024, 2048]
    assert raw["interruption"]["omitted_targets"] == [3072]
    assert raw["interruption"]["completed_requests"] == 12
    assert all(request["compile_markers_in_request"] == 0 for request in raw["requests"])
    assert [aggregate["logical_prompt_tokens"] for aggregate in raw["aggregates"]] == [32, 256, 1024, 2048]
    assert raw["aggregates"][-1]["output_tokens_per_s"] == 16.722168


def test_interrupted_run_is_never_left_running():
    result = {
        "status": "running",
        "requests": [
            {"actual_logical_prompt_tokens": 32},
            {"actual_logical_prompt_tokens": 256},
        ],
    }

    sweep._mark_interrupted(result, KeyboardInterrupt())

    assert result["status"] == "interrupted"
    assert result["interruption"]["completed_requests"] == 2
    assert result["interruption"]["completed_targets"] == [32, 256]
