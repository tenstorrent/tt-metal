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
    PreparedCheckpoint,
    create_arg_parser,
    deterministic_decode_input_ids,
    deterministic_input_ids,
    require_t3k_available,
    run_tiny_model_demo,
    summarize_demo_result,
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

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--tokens", "0"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--warmup-runs", "-1"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--decode-steps", "-1"])
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
