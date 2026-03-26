# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from models.experimental.lingbot_va.tests.demo.inference_ttnn import build_infer_message, run_inference


def _resolve_checkpoint_path() -> Path | None:
    ckpt = os.environ.get("LINGBOT_VA_CHECKPOINT", "").strip()
    if ckpt:
        p = Path(ckpt)
        return p if p.is_dir() else None
    tt_metal_home = os.environ.get("TT_METAL_HOME", "").strip()
    if tt_metal_home:
        p = Path(tt_metal_home) / "models/experimental/lingbot_va/reference/checkpoints"
        return p if p.is_dir() else None
    return None


def _random_obs(seed: int = 42, h: int = 256, w: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cam_high = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    cam_left = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    cam_right = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return cam_high, cam_left, cam_right


def _resolve_perf_save_dir() -> Path:
    override = os.environ.get("LINGBOT_VA_PERF_SAVE_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent / "out_perf_lingbot_va"


@pytest.mark.timeout(0)
def test_lingbot_va():
    checkpoint_path = _resolve_checkpoint_path()
    if checkpoint_path is None:
        pytest.skip("Lingbot checkpoint not found. Set LINGBOT_VA_CHECKPOINT or TT_METAL_HOME.")
    os.environ["LINGBOT_VA_NUM_INFERENCE_STEPS"] = "1"
    os.environ["LINGBOT_VA_ACTION_NUM_INFERENCE_STEPS"] = "1"
    os.environ["LINGBOT_VA_FRAME_CHUNK_SIZE"] = "1"

    cam_high, cam_left, cam_right = _random_obs()
    message = build_infer_message(
        cam_high=cam_high,
        cam_left_wrist=cam_left,
        cam_right_wrist=cam_right,
        prompt="Lift the cup from the table",
    )
    out = run_inference(
        message=message,
        checkpoint_path=checkpoint_path,
        save_dir=_resolve_perf_save_dir(),
    )

    assert "action" in out, "Expected 'action' in run_inference output"
    action = out["action"]
    assert action is not None
    assert getattr(action, "size", 0) > 0
