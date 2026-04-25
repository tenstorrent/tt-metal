# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v4_flash.demo import (
    DEMO_NAME,
    DemoTimings,
    GenerationTimings,
    PreparedCheckpoint,
    create_arg_parser,
    deterministic_decode_input_ids,
    deterministic_input_ids,
    greedy_next_token_id,
    require_t3k_available,
    run_tiny_generation_loop,
    run_tiny_model_demo,
    summarize_demo_result,
    summarize_generation_metrics,
)


def test_tiny_demo_arg_parser_defaults_and_overrides() -> None:
    defaults = create_arg_parser().parse_args([])

    assert defaults.preprocessed_path is None
    assert defaults.artifact_dir is None
    assert defaults.tokens == 32
    assert defaults.layer == 2
    assert defaults.layer_ids is None
    assert defaults.top_k == 5
    assert defaults.warmup_runs == 1
    assert defaults.measure_runs == 1
    assert defaults.decode_steps == 0
    assert defaults.generate_steps == 0

    parsed = create_arg_parser().parse_args(
        [
            "--preprocessed-path",
            "/tmp/deepseek-v4-flash-tt",
            "--artifact-dir",
            "/tmp/deepseek-v4-flash-artifacts",
            "--tokens",
            "64",
            "--layer",
            "1",
            "--layer-ids",
            "0,1",
            "--top-k",
            "3",
            "--warmup-runs",
            "0",
            "--measure-runs",
            "2",
            "--decode-steps",
            "3",
            "--generate-steps",
            "4",
        ]
    )

    assert parsed.preprocessed_path == Path("/tmp/deepseek-v4-flash-tt")
    assert parsed.artifact_dir == Path("/tmp/deepseek-v4-flash-artifacts")
    assert parsed.tokens == 64
    assert parsed.layer == 1
    assert parsed.layer_ids == (0, 1)
    assert parsed.top_k == 3
    assert parsed.warmup_runs == 0
    assert parsed.measure_runs == 2
    assert parsed.decode_steps == 3
    assert parsed.generate_steps == 4

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--tokens", "0"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--warmup-runs", "-1"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--decode-steps", "-1"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--generate-steps", "-1"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--layer-ids", "0,0"])


def test_tiny_demo_deterministic_input_ids() -> None:
    input_ids = deterministic_input_ids(tokens=4, vocab_size=64)

    assert input_ids.shape == (1, 4)
    assert input_ids.dtype == torch.int64
    torch.testing.assert_close(input_ids, torch.zeros(1, 4, dtype=torch.int64))

    with pytest.raises(ValueError, match="tokens must be positive"):
        deterministic_input_ids(tokens=0, vocab_size=64)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        deterministic_input_ids(tokens=4, vocab_size=0)

    decode_input_ids = deterministic_decode_input_ids(tokens=0, vocab_size=64)
    assert decode_input_ids.shape == (1, 0)
    with pytest.raises(ValueError, match="decode tokens must be non-negative"):
        deterministic_decode_input_ids(tokens=-1, vocab_size=64)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        deterministic_decode_input_ids(tokens=0, vocab_size=0)


def test_tiny_demo_summary_reports_shape_checksum_topk_and_timings() -> None:
    logits = torch.tensor(
        [
            [
                [0.5, -1.0, 2.0, 1.0],
                [3.0, 0.0, -0.25, 1.25],
            ]
        ],
        dtype=torch.float32,
    )
    input_ids = torch.zeros(1, 2, dtype=torch.int64)
    summary = summarize_demo_result(
        logits=logits,
        input_ids=input_ids,
        timings=DemoTimings(setup_s=0.1, model_init_s=0.2, warmup_s=0.3, run_s=0.4, total_s=1.0),
        checkpoint=PreparedCheckpoint(Path("/tmp/tt_preprocessed"), generated_synthetic_checkpoint=False),
        layer=2,
        top_k=3,
        warmup_runs=1,
        measure_runs=2,
    )

    assert summary["demo"] == DEMO_NAME
    assert "Tiny scaffold/demo only" in summary["note"]
    assert summary["checkpoint"] == {
        "preprocessed_path": "/tmp/tt_preprocessed",
        "generated_synthetic_checkpoint": False,
    }
    assert summary["input"]["token_count"] == 2
    assert summary["input"]["layer"] == 2
    assert summary["input"]["layer_ids"] == [2]
    assert summary["input"]["warmup_runs"] == 1
    assert summary["input"]["measure_runs"] == 2
    assert summary["input"]["decode_steps"] == 0
    assert summary["input"]["generate_steps"] == 0
    assert summary["logits"]["shape"] == [1, 2, 4]
    assert summary["logits"]["checksum"] == 6.5
    assert summary["logits"]["first_token_top_k"] == [
        {"id": 2, "value": 2.0},
        {"id": 3, "value": 1.0},
        {"id": 0, "value": 0.5},
    ]
    assert summary["timing_s"] == {
        "setup": 0.1,
        "model_init": 0.2,
        "warmup": 0.3,
        "run": 0.4,
        "total": 1.0,
    }


def test_tiny_demo_summary_reports_decode_logits_and_timing() -> None:
    logits = torch.tensor([[[0.5, 1.0, -0.5, 0.25], [0.0, -1.0, 2.0, 1.0]]], dtype=torch.float32)
    decode_logits = torch.tensor([[[1.0, -1.0, 0.0, 0.5], [0.25, 3.0, 2.5, -0.5]]], dtype=torch.float32)
    input_ids = torch.zeros(1, 2, dtype=torch.int64)
    summary = summarize_demo_result(
        logits=logits,
        decode_logits=decode_logits,
        input_ids=input_ids,
        timings=DemoTimings(setup_s=0.1, model_init_s=0.2, warmup_s=0.3, run_s=0.7, total_s=1.2, decode_s=0.25),
        checkpoint=PreparedCheckpoint(Path("/tmp/tt_preprocessed"), generated_synthetic_checkpoint=False),
        layer_ids=(1, 2),
        top_k=2,
        warmup_runs=0,
        measure_runs=1,
        decode_steps=2,
    )

    assert summary["input"]["decode_steps"] == 2
    assert summary["input"]["layer_ids"] == [1, 2]
    assert summary["decode_logits"]["shape"] == [1, 2, 4]
    assert summary["decode_logits"]["checksum"] == 5.75
    assert summary["decode_logits"]["final_token_top_k"] == [{"id": 1, "value": 3.0}, {"id": 2, "value": 2.5}]
    assert summary["timing_s"]["decode"] == 0.25


def test_greedy_next_token_id_selects_last_token_argmax_deterministically() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 1.0, 2.0],
                [0.5, 2.0, 2.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    )

    next_token = greedy_next_token_id(logits)

    assert next_token.shape == (1, 1)
    assert next_token.dtype == torch.int64
    assert int(next_token.item()) == 1

    with pytest.raises(ValueError, match=r"logits must have shape"):
        greedy_next_token_id(torch.zeros(1, 4))
    with pytest.raises(ValueError, match="at least one token"):
        greedy_next_token_id(torch.zeros(1, 0, 4))
    with pytest.raises(ValueError, match="at least one vocab"):
        greedy_next_token_id(torch.zeros(1, 1, 0))


def test_generation_metrics_math() -> None:
    metrics = summarize_generation_metrics(
        prompt_tokens=8,
        generated_tokens=4,
        timings=GenerationTimings(prefill_s=0.2, decode_s=0.5, total_s=0.8),
    )

    assert metrics == {
        "prompt_tokens": 8,
        "generated_tokens": 4,
        "users": 1,
        "prefill_latency_s": 0.2,
        "total_decode_latency_s": 0.5,
        "per_token_decode_latency_s": 0.125,
        "generation_total_latency_s": 0.8,
        "effective_decode_tokens_s_per_user": 8.0,
    }

    zero_metrics = summarize_generation_metrics(
        prompt_tokens=8,
        generated_tokens=0,
        timings=GenerationTimings(prefill_s=0.2, decode_s=0.0, total_s=0.3),
    )
    assert zero_metrics["per_token_decode_latency_s"] == 0.0
    assert zero_metrics["effective_decode_tokens_s_per_user"] == 0.0

    with pytest.raises(ValueError, match="prompt_tokens must be positive"):
        summarize_generation_metrics(
            prompt_tokens=0,
            generated_tokens=1,
            timings=GenerationTimings(prefill_s=0.0, decode_s=0.1, total_s=0.1),
        )
    with pytest.raises(ValueError, match="generated_tokens must be non-negative"):
        summarize_generation_metrics(
            prompt_tokens=1,
            generated_tokens=-1,
            timings=GenerationTimings(prefill_s=0.0, decode_s=0.1, total_s=0.1),
        )


def test_tiny_generation_loop_uses_greedy_tokens_and_reports_timings() -> None:
    class _Cache:
        def __init__(self, current_position: int):
            self.current_position = current_position

    class _StepTimer:
        def __init__(self):
            self.value = -0.1

        def __call__(self) -> float:
            self.value += 0.1
            return self.value

    class _FakeModel:
        def __init__(self):
            self.decode_inputs = []

        def prefill_with_decode_cache(self, input_ids):
            logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 3.0, 2.0]]], dtype=torch.float32)
            return logits, _Cache(current_position=int(input_ids.shape[1]))

        def decode_step(self, input_ids, *, cache):
            self.decode_inputs.append(input_ids.clone())
            if len(self.decode_inputs) == 1:
                logits = torch.tensor([[[0.0, 4.0, 1.0, 2.0]]], dtype=torch.float32)
            else:
                logits = torch.tensor([[[0.0, 1.0, 2.0, 5.0]]], dtype=torch.float32)
            return logits, _Cache(current_position=cache.current_position + 1)

    sync_count = 0

    def synchronize():
        nonlocal sync_count
        sync_count += 1

    model = _FakeModel()
    result = run_tiny_generation_loop(
        model,
        input_ids=torch.zeros(1, 2, dtype=torch.int64),
        generate_steps=2,
        synchronize=synchronize,
        timer=_StepTimer(),
    )

    assert result.generated_token_ids.tolist() == [[2, 1]]
    assert [int(token.item()) for token in model.decode_inputs] == [2, 1]
    assert result.decode_logits.shape == (1, 2, 4)
    assert result.cache_current_position == 4
    assert sync_count == 3
    assert result.timings.prefill_s == pytest.approx(0.1)
    assert result.timings.decode_s == pytest.approx(0.2)
    assert result.timings.total_s == pytest.approx(0.7)


@pytest.mark.t3k_compat
def test_t3k_tiny_model_demo_direct_smoke(tmp_path) -> None:
    ttnn = pytest.importorskip("ttnn")
    try:
        require_t3k_available(ttnn)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    args = create_arg_parser().parse_args(
        [
            "--artifact-dir",
            str(tmp_path),
            "--warmup-runs",
            "0",
            "--measure-runs",
            "1",
            "--top-k",
            "3",
        ]
    )

    summary = run_tiny_model_demo(args)

    assert summary["demo"] == DEMO_NAME
    assert summary["checkpoint"]["generated_synthetic_checkpoint"] is True
    assert Path(summary["checkpoint"]["preprocessed_path"]).is_dir()
    assert summary["input"]["token_count"] == 32
    assert summary["logits"]["shape"] == [1, 32, 64]
    assert len(summary["logits"]["first_token_top_k"]) == 3
    assert summary["timing_s"]["run"] > 0.0


@pytest.mark.t3k_compat
def test_t3k_tiny_model_demo_decode_smoke(tmp_path) -> None:
    ttnn = pytest.importorskip("ttnn")
    try:
        require_t3k_available(ttnn)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    args = create_arg_parser().parse_args(
        [
            "--artifact-dir",
            str(tmp_path),
            "--tokens",
            "16",
            "--warmup-runs",
            "0",
            "--measure-runs",
            "1",
            "--top-k",
            "3",
            "--decode-steps",
            "2",
        ]
    )

    summary = run_tiny_model_demo(args)

    assert summary["demo"] == DEMO_NAME
    assert summary["input"]["token_count"] == 16
    assert summary["input"]["decode_steps"] == 2
    assert summary["logits"]["shape"] == [1, 16, 64]
    assert summary["decode_logits"]["shape"] == [1, 2, 64]
    assert len(summary["decode_logits"]["final_token_top_k"]) == 3
    assert summary["timing_s"]["decode"] > 0.0


@pytest.mark.t3k_compat
def test_t3k_tiny_model_demo_generation_smoke(tmp_path) -> None:
    ttnn = pytest.importorskip("ttnn")
    try:
        require_t3k_available(ttnn)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    args = create_arg_parser().parse_args(
        [
            "--artifact-dir",
            str(tmp_path),
            "--tokens",
            "16",
            "--warmup-runs",
            "0",
            "--measure-runs",
            "1",
            "--top-k",
            "3",
            "--generate-steps",
            "2",
        ]
    )

    summary = run_tiny_model_demo(args)

    assert summary["demo"] == DEMO_NAME
    assert summary["input"]["token_count"] == 16
    assert summary["input"]["decode_steps"] == 0
    assert summary["input"]["generate_steps"] == 2
    assert summary["logits"]["shape"] == [1, 16, 64]
    assert summary["generation"]["generated_token_ids"]
    assert len(summary["generation"]["generated_token_ids"]) == 2
    assert summary["generation"]["decode_cache_current_position"] == 18
    assert summary["generation"]["decode_logits"]["shape"] == [1, 2, 64]
    assert len(summary["generation"]["decode_logits"]["final_token_top_k"]) == 3
    assert summary["generation"]["metrics"]["prompt_tokens"] == 16
    assert summary["generation"]["metrics"]["generated_tokens"] == 2
    assert summary["generation"]["metrics"]["prefill_latency_s"] > 0.0
    assert summary["generation"]["metrics"]["total_decode_latency_s"] > 0.0
    assert summary["generation"]["metrics"]["per_token_decode_latency_s"] > 0.0
    assert summary["generation"]["metrics"]["effective_decode_tokens_s_per_user"] > 0.0
