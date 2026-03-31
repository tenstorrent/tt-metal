# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone runner: TTNN ``demo.run_inference`` for Tracy nested pytest."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=r".*SwigPy(Packed|Object).*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*swigvarlink.*", category=DeprecationWarning)

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
    frame_chunk_size: int = 1
    prompt: str = "Lift the cup from the table"
    obs_seed: int = 42
    obs_h: int = 256
    obs_w: int = 256


def create_config() -> LingbotVaInferenceConfig:
    """Create Lingbot-VA test config (fast single-step run)."""
    return LingbotVaInferenceConfig()


def create_test_inputs(config: LingbotVaInferenceConfig, batch_size: int = 1):
    """Build observation arrays and the infer ``message`` dict."""
    if batch_size != 1:
        raise ValueError("Lingbot VA run_inference supports batch_size=1.")
    rng = np.random.default_rng(config.obs_seed)
    cam_high = rng.integers(0, 256, size=(config.obs_h, config.obs_w, 3), dtype=np.uint8)
    cam_left = rng.integers(0, 256, size=(config.obs_h, config.obs_w, 3), dtype=np.uint8)
    cam_right = rng.integers(0, 256, size=(config.obs_h, config.obs_w, 3), dtype=np.uint8)
    message = lingbot_demo.build_infer_message(
        cam_high=cam_high,
        cam_left_wrist=cam_left,
        cam_right_wrist=cam_right,
        prompt=config.prompt,
    )
    return {
        "message": message,
        "cam_high": cam_high,
        "cam_left": cam_left,
        "cam_right": cam_right,
    }


def _inference_kwargs_from_config(config: LingbotVaInferenceConfig) -> dict:
    """Keyword args for ``demo.run_inference`` / ``inference_torch.run_inference``."""
    return {
        "num_inference_steps": config.num_inference_steps,
        "action_num_inference_steps": config.action_num_inference_steps,
        "frame_chunk_size": config.frame_chunk_size,
    }


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
    kw = _inference_kwargs_from_config(config)

    out = lingbot_demo.run_inference(
        message=inputs["message"],
        checkpoint_path=checkpoint_path,
        **kw,
    )

    assert "action" in out, "Expected 'action' in run_inference output"
    assert out["action"] is not None
    assert getattr(out["action"], "size", 0) > 0


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(SwigPy|swigvarlink).*:DeprecationWarning",
)


@pytest.mark.timeout(600)
def test_lingbot_va_ttnn_forward_run():
    """TTNN-only forward entrypoint for nested profiler subprocess."""
    _run_lingbot_va_ttnn_forward()
