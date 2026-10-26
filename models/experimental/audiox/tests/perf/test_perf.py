# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end perf test for the AudioX bringup.

The dominant cost of a generation is the DiT denoiser: AudioX runs ~50
sampler steps × 24 transformer blocks per step. The Oobleck decoder runs
once and is much cheaper per generation; conditioners are <0.1% of total.
So this test focuses on the DiT step latency and reports per-generation
estimates derived from it.

Run on N300 hardware:

    pytest models/experimental/audiox/tests/perf/test_perf.py \
           -k models_performance_bare_metal -s
"""

import time

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.audiox.reference.dit import DiffusionTransformer
from models.experimental.audiox.tt.dit import TtDiffusionTransformer


# AudioX HF config dims; pinned here to keep the perf test self-contained.
_HF_CONFIG = {
    "io_channels": 64,
    "embed_dim": 1536,
    "depth": 24,
    "num_heads": 24,
    "cond_token_dim": 768,
    "t_latent": 237,  # 10s @ 44.1kHz / 2048 downsample (rounded up from 216)
    "cond_seq_len": 346,  # 128 (T5) + 128 (CLIP) + 90 (audio empty); approx upper bound
    "sampler_steps": 50,
}

# Targets — initial guesses informed by DiT-S/L precedents on N300; tighten
# once we have real numbers in hand.
_TARGET_STEP_MS = 200.0
_TARGET_GEN_S = 12.0


def _build_tt_dit(device) -> TtDiffusionTransformer:
    """Build the TT DiT seeded from a random reference module's state_dict.
    For perf we don't need pretrained weights — only the shapes matter."""
    torch.manual_seed(0)
    ref = DiffusionTransformer(
        io_channels=_HF_CONFIG["io_channels"],
        embed_dim=_HF_CONFIG["embed_dim"],
        depth=_HF_CONFIG["depth"],
        num_heads=_HF_CONFIG["num_heads"],
        cond_token_dim=_HF_CONFIG["cond_token_dim"],
    ).eval()
    return TtDiffusionTransformer(
        mesh_device=device,
        state_dict=ref.state_dict(),
        depth=_HF_CONFIG["depth"],
        num_heads=_HF_CONFIG["num_heads"],
        io_channels=_HF_CONFIG["io_channels"],
        embed_dim=_HF_CONFIG["embed_dim"],
    )


def _make_inputs(device):
    x = torch.randn(1, _HF_CONFIG["io_channels"], _HF_CONFIG["t_latent"])
    t = torch.tensor([0.5])
    cond = torch.randn(1, _HF_CONFIG["cond_seq_len"], _HF_CONFIG["cond_token_dim"])
    return (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def _benchmark_step(model, inputs, *, warmup: int, iters: int) -> float:
    """Mean per-step latency in ms, after `warmup` discarded passes."""
    x, t, cond = inputs
    for _ in range(warmup):
        _ = model(x, t, cross_attn_cond=cond)
    ttnn.synchronize_device(x.device())

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x, t, cross_attn_cond=cond)
    ttnn.synchronize_device(x.device())
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


@pytest.mark.models_performance_bare_metal
def test_audiox_dit_step_latency(device):
    """One DiT forward must come in under the per-step budget so a 50-step
    generation fits inside the per-generation target."""
    model = _build_tt_dit(device)
    inputs = _make_inputs(device)

    step_ms = _benchmark_step(model, inputs, warmup=2, iters=10)
    gen_s = (step_ms * _HF_CONFIG["sampler_steps"]) / 1000.0

    logger.info(f"DiT step: {step_ms:.2f} ms (target < {_TARGET_STEP_MS} ms)")
    logger.info(
        f"Estimated generation: {gen_s:.2f} s for {_HF_CONFIG['sampler_steps']} steps (target < {_TARGET_GEN_S} s)"
    )

    assert step_ms < _TARGET_STEP_MS, f"DiT step {step_ms:.2f} ms exceeds target {_TARGET_STEP_MS} ms"
    assert gen_s < _TARGET_GEN_S, f"Estimated generation {gen_s:.2f} s exceeds target {_TARGET_GEN_S} s"
