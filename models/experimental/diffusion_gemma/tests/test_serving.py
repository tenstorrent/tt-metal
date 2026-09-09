# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical serving lifecycle, row-contract, and device regressions."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

serving = pytest.importorskip("models.experimental.diffusion_gemma.tt.serving")

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
DG_CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def test_session_reset_releases_eager_logits_state():
    events = []
    logits_fn = SimpleNamespace(reset=lambda: events.append("logits-reset"))
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = logits_fn
    session._persistent_adapter = None
    session.next_pos = 288
    session.finished = True
    session.block_idx = 2

    session.reset()

    assert events == ["logits-reset"]
    assert session._logits_fn is None
    assert session._persistent_adapter is None
    assert session.next_pos is None
    assert session.finished is False
    assert session.block_idx == 0


def test_next_block_capacity_accepts_exact_boundary_after_nonaligned_prompt():
    model = SimpleNamespace(max_seq_len=544)
    serving._validate_next_block_capacity(model, start_pos=288, canvas_length=256)


def test_decode_rejects_block_overrun_before_device_execution(monkeypatch, expect_error):
    device_called = False

    def _unexpected_device_call(*args, **kwargs):
        nonlocal device_called
        device_called = True
        raise AssertionError("device execution must not begin")

    monkeypatch.setattr(serving, "denoise_and_commit_block", _unexpected_device_call)
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = object()
    session.next_pos = 320
    session.finished = False
    session.tt_model = SimpleNamespace(max_seq_len=544)
    session.canvas_length = 256

    with expect_error(ValueError, match=r"320 \+ 256 = 576 > 544"):
        session.decode_block()
    assert device_called is False


class _TerminalEmission:
    def __init__(self):
        self.tokens = torch.zeros(0, dtype=torch.long)
        self.block_idx = 3


class _SessionStub:
    stop_token_ids = []


def test_terminal_emission_uses_tokenizer_eos_when_vllm_stop_policy_is_empty():
    generator_vllm = pytest.importorskip("models.experimental.diffusion_gemma.tt.generator_vllm")
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.canvas_length = 256
    wrapper._tokenizer = SimpleNamespace(eos_token_id=[106, 1])

    block = wrapper._emission_block(_TerminalEmission(), _SessionStub(), row=0)

    assert block.shape == (1, 256)
    assert (block == 106).all()


@pytest.mark.skipif(not DEVICE_GATED, reason="device serving smoke requires DG_RUN_DEVICE=1")
@pytest.mark.skipif(not os.path.isdir(DG_CKPT), reason=f"checkpoint not available at {DG_CKPT}")
def test_serving_smoke_emits_blocks_and_advances_position():
    from models.experimental.diffusion_gemma.tests.serving_smoke import build_arg_parser, run

    canvas = 256
    args = build_arg_parser().parse_args(
        [
            "--checkpoint",
            DG_CKPT,
            "--mesh",
            os.environ.get("DG_MESH", "P150x4"),
            "--num-layers",
            os.environ.get("DG_VLLM_SMOKE_NUM_LAYERS", "1"),
            "--max-seq-len",
            "1024",
            "--num-blocks",
            "2",
            "--canvas-length",
            str(canvas),
            "--max-denoising-steps",
            os.environ.get("DG_VLLM_SMOKE_STEPS", "2"),
            "--gumbel-mode",
            os.environ.get("DG_VLLM_SMOKE_GUMBEL", "argmax"),
            "--local-files-only",
        ]
    )
    metrics = run(args)

    assert metrics["canvas_length"] == canvas
    assert metrics["blocks_emitted"] >= 1
    assert metrics["tokens_emitted"] == metrics["blocks_emitted"] * canvas
    assert metrics["prompt_aligned_256"] is False
    assert metrics["final_next_pos"] == metrics["cache_len"] + metrics["blocks_emitted"] * canvas
    assert metrics["ttft_s"] > 0.0
    assert metrics["mean_block_latency_s"] > 0.0
    assert metrics["tokens_per_block_per_s"] > 0.0
    assert len(metrics["per_block_latency_s"]) == metrics["blocks_emitted"]
