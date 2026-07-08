# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-capture bring-up / debug for the BH Galaxy pipeline (Phase B.3).

Exercises Pi0_5GLXPipeline.capture_trace + sample_actions_traced. The warmup
inside capture_trace is the pass that historically hangs; run with
PI0_GLX_TRACE_DEBUG=1 to print per-stage markers + drain each stage's submesh
so a stall localizes to a specific stage boundary.

Opt-in only (needs a 32-chip Galaxy and may hang): set PI0_GLX_TRACE_TEST=1.

    PI0_GLX_TRACE_TEST=1 PI0_GLX_TRACE_DEBUG=1 \
      PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
      python_env/bin/pytest -xvs \
      models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_trace.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh


CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42


@pytest.mark.skipif(
    os.environ.get("PI0_GLX_TRACE_TEST", "").lower() not in ("1", "true", "yes", "on"),
    reason="opt-in only (32-chip Galaxy, may hang); set PI0_GLX_TRACE_TEST=1",
)
@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_pi0_5_glx_capture_trace():
    """Build the pipeline, capture a trace, replay it once, sanity-check output."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline import Pi0_5GLXPipeline

    num_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
    num_cams = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=num_steps,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))

    img_h = img_w = cfg.siglip_config.image_size
    lang_len = 256
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, img_h, img_w) for _ in range(num_cams)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cams)]
    lang_tokens = torch.randint(0, 256000, (1, lang_len), dtype=torch.int32)
    lang_masks = torch.ones(1, lang_len, dtype=torch.bool)

    print(f"\n📋 GLX capture_trace — cams={num_cams}, denoise_steps={num_steps}, ckpt={CHECKPOINT_DIR}")

    with open_galaxy_mesh(l1_small_size=24576) as h:
        pipe = Pi0_5GLXPipeline(cfg, loader.categorized_weights, h)

        print("\n🎬 capture_trace (warmup is the historically-hanging pass)")
        pipe.capture_trace(images, img_masks, lang_tokens, lang_masks)
        print("✅ capture_trace returned")

        print("\n▶️  sample_actions_traced")
        actions = pipe.sample_actions_traced(images, lang_tokens)
        print(f"✅ traced actions shape={tuple(actions.shape)}")
        assert torch.isfinite(actions).all(), "traced actions contain NaN/Inf"
