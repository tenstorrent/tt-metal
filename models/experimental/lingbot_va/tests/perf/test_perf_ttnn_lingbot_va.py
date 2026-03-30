# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device perf: Tracy profile of TTNN ``demo.run_inference`` (nested pytest on ``test_lingbot_va_ttnn_forward_run``).

Inference helpers below mirror ``tests/pcc/test_lingbot_va.py`` so this module does not import PCC tests.
"""

from __future__ import annotations

import warnings

# Before any import that may pull ttnn/SWIG (so Python does not emit these during import).
warnings.filterwarnings("ignore", message=r".*SwigPy(Packed|Object).*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*swigvarlink.*", category=DeprecationWarning)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from tracy.process_model_log import run_device_profiler

import models
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo

# =============================================================================
# CONFIG (aligned with tests/pcc/test_lingbot_va.py; keep in sync manually)
# =============================================================================
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


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(SwigPy|swigvarlink).*:DeprecationWarning",
)


def _run_device_profiler_op_support_count(*args, **kwargs):
    # Default cap is low; Lingbot-VA graphs exceed it without this override.
    kwargs.setdefault("op_support_count", 7500)
    return run_device_profiler(*args, **kwargs)


models.perf.device_perf_utils.run_device_profiler = _run_device_profiler_op_support_count


def _run_lingbot_va_ttnn_forward() -> None:
    """Single TTNN ``demo.run_inference`` (one chunk); writes under ``out_perf_lingbot_va_forward``."""
    checkpoint_path = "models/experimental/lingbot_va/reference/checkpoints"

    config = create_config()
    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
    kw = _inference_kwargs_from_config(config)

    save_dir = Path(__file__).resolve().parent / "out_perf_lingbot_va_forward"
    save_dir.mkdir(parents=True, exist_ok=True)

    out = lingbot_demo.run_inference(
        message=inputs["message"],
        checkpoint_path=checkpoint_path,
        save_dir=save_dir,
        **kw,
    )

    assert "action" in out, "Expected 'action' in run_inference output"
    assert out["action"] is not None
    assert getattr(out["action"], "size", 0) > 0


@pytest.mark.timeout(0)
def test_lingbot_va_ttnn_forward_run():
    """TTNN-only forward for Tracy (nested by ``test_perf_device_bare_metal_lingbot_va``)."""
    _run_lingbot_va_ttnn_forward()


@pytest.mark.parametrize(
    "batch_size, model_name",
    [
        (1, "ttnn_lingbot_va"),
    ],
)
@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_lingbot_va(batch_size, model_name):
    subdir = model_name
    num_iterations = 1
    margin = 0.16

    command = "pytest models/experimental/lingbot_va/tests/perf/test_perf_ttnn_lingbot_va.py::test_lingbot_va_ttnn_forward_run"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    # Baseline samples/s for one forward (device-dependent); ±margin must bracket Tracy post-process output.
    expected_perf_cols = {inference_time_key: 0.49}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
