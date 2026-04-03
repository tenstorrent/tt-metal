# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone runner: TTNN ``demo.run_inference`` for Tracy nested pytest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo

# Same layout as ``test_perf_e2e.py``: ``tests/perf`` → six parents up to tt-metal repo root.
_TT_METAL_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
CHECKPOINT_RELATIVE_TO_REPO = "models/experimental/lingbot_va/reference/checkpoints"

BATCH_SIZE = 1


@dataclass
class LingbotVaInferenceConfig:
    """Inference settings passed through to ``run_inference`` (and obs / prompt for inputs)."""

    num_inference_steps: int = 1
    action_num_inference_steps: int = 1
    # Match ``VA_CONFIGS["robotwin"].frame_chunk_size`` (6). ``frame_chunk_size=1`` hits a fragile
    # per-token timestep concat path on device; CLI ``demo.py`` uses the same default when kwargs omitted.
    frame_chunk_size: int = 6
    prompt: str = "Lift the cup from the table"
    obs_seed: int = 42
    obs_h: int = 256
    obs_w: int = 256


def create_config() -> LingbotVaInferenceConfig:
    """Create Lingbot-VA test config (fast single-step run)."""
    return LingbotVaInferenceConfig()


def create_test_inputs(config: LingbotVaInferenceConfig, batch_size: int = 1):
    """Build random camera frames and the infer ``message`` dict for ``run_inference``."""
    if batch_size != 1:
        raise ValueError("Lingbot VA run_inference supports batch_size=1.")
    rng = np.random.default_rng(config.obs_seed)
    cam_high, cam_left, cam_right = (
        rng.integers(0, 256, size=(config.obs_h, config.obs_w, 3), dtype=np.uint8) for _ in range(3)
    )
    message = lingbot_demo.build_infer_message(
        cam_high=cam_high,
        cam_left_wrist=cam_left,
        cam_right_wrist=cam_right,
        prompt=config.prompt,
    )
    return {"message": message}


def _run_lingbot_va_ttnn_forward() -> None:
    """Single TTNN ``demo.run_inference`` (one chunk); writes under ``out_perf_lingbot_va_forward``."""
    # Absolute path: ``run_inference`` resolves relative paths against *process* cwd; ``demo`` also
    # ``chdir``s to the Lingbot package root, so a relative string can double-resolve after another
    # test run (e.g. ``.../lingbot_va/models/experimental/lingbot_va/...``).
    checkpoint_path = _TT_METAL_ROOT / CHECKPOINT_RELATIVE_TO_REPO
    if not checkpoint_path.is_dir():
        pytest.skip(f"Checkpoint dir not found: {checkpoint_path}")

    config = create_config()
    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)

    out = lingbot_demo.run_inference(
        message=inputs["message"],
        checkpoint_path=checkpoint_path,
        num_inference_steps=config.num_inference_steps,
        action_num_inference_steps=config.action_num_inference_steps,
        frame_chunk_size=config.frame_chunk_size,
    )

    assert isinstance(out, dict), "run_inference returns {'action': ndarray}"
    assert "action" in out, "Expected 'action' in run_inference output"
    assert out["action"] is not None
    assert getattr(out["action"], "size", 0) > 0


@pytest.mark.timeout(600)
def test_lingbot_va_ttnn_forward_run():
    """TTNN-only forward entrypoint for nested profiler subprocess."""
    _run_lingbot_va_ttnn_forward()
