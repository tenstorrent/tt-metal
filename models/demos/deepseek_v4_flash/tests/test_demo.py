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
    assert defaults.top_k == 5
    assert defaults.warmup_runs == 1
    assert defaults.measure_runs == 1

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
            "--top-k",
            "3",
            "--warmup-runs",
            "0",
            "--measure-runs",
            "2",
        ]
    )

    assert parsed.preprocessed_path == Path("/tmp/deepseek-v4-flash-tt")
    assert parsed.artifact_dir == Path("/tmp/deepseek-v4-flash-artifacts")
    assert parsed.tokens == 64
    assert parsed.layer == 1
    assert parsed.top_k == 3
    assert parsed.warmup_runs == 0
    assert parsed.measure_runs == 2

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--tokens", "0"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        create_arg_parser().parse_args(["--warmup-runs", "-1"])


def test_tiny_demo_deterministic_input_ids() -> None:
    input_ids = deterministic_input_ids(tokens=4, vocab_size=64)

    assert input_ids.shape == (1, 4)
    assert input_ids.dtype == torch.int64
    torch.testing.assert_close(input_ids, torch.zeros(1, 4, dtype=torch.int64))

    with pytest.raises(ValueError, match="tokens must be positive"):
        deterministic_input_ids(tokens=0, vocab_size=64)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        deterministic_input_ids(tokens=4, vocab_size=0)


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
    assert summary["input"]["warmup_runs"] == 1
    assert summary["input"]["measure_runs"] == 2
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
