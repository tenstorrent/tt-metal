# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving block-contract tests for the DiffusionGemma vLLM adapter (#47466).

CPU tests cover the vLLM-free block-emission scaffolding (sampler modes, argmax
Gumbel hook, session validation) that ``tt/generator_vllm.py`` delegates to.

The device test (``DG_RUN_DEVICE=1``) runs the reduced-surface serving driver
``demo/serving_smoke.py`` — prefill + N committed 256-token blocks with a
non-256-aligned prompt — and asserts the block-granular contract: number of
blocks, tokens emitted, position advancement by ``canvas_length``, and that a
per-block metrics dict (TTFT, per-block latency, tokens-per-block) was produced.
RUN-first: text quality is NOT gated (degenerate output expected until #48291).
"""

from __future__ import annotations

import os

import pytest

from models.experimental.diffusion_gemma.tt import serving

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
DG_CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


# ── CPU: vLLM-free scaffolding ──────────────────────────────────────────
def test_gumbel_modes_cover_argmax_and_chunked():
    # argmax + chunked are the two that fit full-depth 256K per the context contract.
    assert "argmax" in serving.GUMBEL_MODES
    assert "chunked" in serving.GUMBEL_MODES


def test_argmax_gumbel_hook_returns_none_per_step():
    per_step = serving._argmax_gumbel_noise_fn(0)
    assert per_step(0) is None
    assert per_step(5) is None


def test_argmax_gumbel_hook_rejects_bad_block_index():
    with pytest.raises(ValueError):
        serving._argmax_gumbel_noise_fn(True)  # bool is not a valid block index


def test_block_emission_fields():
    assert serving.BlockEmission._fields == (
        "tokens",
        "block_idx",
        "start_pos",
        "next_pos",
        "num_denoise_steps",
        "halted",
        "stop",
        "latency_s",
    )


def test_session_rejects_unknown_gumbel_mode():
    class _M:
        mesh_device = None
        hf_config = None

    with pytest.raises(ValueError):
        serving.BlockDiffusionServingSession(_M(), {}, vocab_size=262144, gumbel_mode="nope")


def test_session_requires_vocab_size_source():
    class _M:
        mesh_device = None
        hf_config = None
        vocab_size = None

    with pytest.raises(ValueError):
        # No tokenizer, no vocab_size, no model vocab metadata → must raise.
        serving.BlockDiffusionServingSession(_M(), {}, gumbel_mode="argmax")


# ── Device: block-granular emission through the serving driver ──────────
@pytest.mark.skipif(not DEVICE_GATED, reason="device serving smoke requires DG_RUN_DEVICE=1")
@pytest.mark.skipif(not os.path.isdir(DG_CKPT), reason=f"checkpoint not available at {DG_CKPT}")
def test_serving_smoke_emits_blocks_and_advances_position():
    from models.experimental.diffusion_gemma.demo.serving_smoke import build_arg_parser, run

    num_layers = os.environ.get("DG_VLLM_SMOKE_NUM_LAYERS", "1")
    canvas = 256
    argv = [
        "--checkpoint",
        DG_CKPT,
        "--mesh",
        os.environ.get("DG_MESH", "P150x4"),
        "--num-layers",
        num_layers,
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
    args = build_arg_parser().parse_args(argv)
    metrics = run(args)

    # Block-granular contract assertions (NOT text quality — RUN-first).
    assert metrics["canvas_length"] == canvas
    assert metrics["blocks_emitted"] >= 1
    assert metrics["tokens_emitted"] == metrics["blocks_emitted"] * canvas
    # Non-aligned prompt carve-out: prompt length is not a multiple of 256.
    assert metrics["prompt_aligned_256"] is False
    # Position advanced by canvas_length per emitted block from the aligned cache_len.
    assert metrics["final_next_pos"] == metrics["cache_len"] + metrics["blocks_emitted"] * canvas
    # Per-block metrics present.
    assert metrics["ttft_s"] > 0.0
    assert metrics["mean_block_latency_s"] > 0.0
    assert metrics["tokens_per_block_per_s"] > 0.0
    assert len(metrics["per_block_latency_s"]) == metrics["blocks_emitted"]
