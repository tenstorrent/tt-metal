# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic request metadata and comparison with an independent CUDA run."""

import json
import os
from pathlib import Path

import pytest
import torch

from models.experimental.qwen_image_2_1.tt.inference_metadata import build_metadata


@pytest.fixture
def configuration_only_checkpoint(tmp_path):
    # These are the pinned upstream configs; there are deliberately no weight files.
    transformer = {
        "_class_name": "QwenImage21Transformer2DModel",
        "patch_size": 1,
        "in_channels": 64,
        "axes_dims_rope": [16, 56, 56],
    }
    scheduler = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "base_shift": 0.5,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "use_dynamic_shifting": True,
        "time_shift_type": "exponential",
    }
    (tmp_path / "transformer").mkdir()
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "transformer/config.json").write_text(json.dumps(transformer))
    (tmp_path / "scheduler/scheduler_config.json").write_text(json.dumps(scheduler))
    return tmp_path


@pytest.mark.parametrize("height,width,steps,prefix", [(256, 384, 20, 17), (384, 256, 40, 31), (32, 32, 2, 1)])
def test_metadata_without_weights(configuration_only_checkpoint, height, width, steps, prefix):
    result = build_metadata(configuration_only_checkpoint, prefix, height, width, steps)
    targets = height // 16 * (width // 16)
    assert result["rope"].shape == (prefix + targets, 64)
    assert result["rope"].device.type == "cpu"
    assert torch.allclose(result["rope"].abs(), torch.ones(prefix + targets, 64), atol=1e-7)
    assert not result["target_mask"][:prefix].any()
    assert result["target_mask"][prefix:].all()
    assert result["segments"] == [(0, prefix, True)]
    assert result["key_valid"] is None
    sigmas = torch.tensor(result["schedule"]["sigmas"])
    assert len(sigmas) == steps + 1
    assert sigmas[-1] == 0
    assert (sigmas[:-1] > sigmas[1:]).all()
    assert result["schedule"]["timesteps"] == (sigmas[:-1].mul(1000).to(torch.bfloat16) / 1000).tolist()


@pytest.mark.parametrize(
    "prefix,height,width,steps",
    [
        (0, 256, 384, 20),
        (17, 255, 384, 20),
        (17, 256, 16, 20),
        (17, 256, 384, 0),
        (17, 256, 384, 1),
        (8192, 256, 384, 20),
        (17, 256, 384, True),
    ],
)
def test_invalid_request(configuration_only_checkpoint, prefix, height, width, steps):
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only test also runs without tt-metal hardware fixtures
        build_metadata(configuration_only_checkpoint, prefix, height, width, steps)


def test_metadata_against_cuda_capture():
    checkpoint = os.environ.get("QWEN_IMAGE21_METADATA_CHECKPOINT")
    capture = os.environ.get("QWEN_IMAGE21_METADATA_CAPTURE")
    if not checkpoint and not capture:
        pytest.skip("set QWEN_IMAGE21_METADATA_CHECKPOINT and QWEN_IMAGE21_METADATA_CAPTURE")
    if not checkpoint or not capture:
        pytest.fail("metadata checkpoint and capture must be set together")
    capture = Path(capture)
    manifest = json.loads((capture / "manifest.json").read_text())
    first = capture / "step_000/transformer/transformer_blocks.0/input"
    mask = torch.load(first / "target_token_mask.pt", map_location="cpu", weights_only=True)
    result = build_metadata(
        Path(checkpoint), int((~mask).sum()), manifest["height"], manifest["width"], manifest["steps"]
    )
    assert torch.equal(result["target_mask"], mask)
    assert torch.equal(result["rope"], torch.load(first / "rotary_emb.pt", map_location="cpu", weights_only=True))
    assert result["segments"] == [tuple(segment) for segment in json.loads((first / "segments.json").read_text())]
    schedule = json.loads((capture / "schedule.json").read_text())
    torch.testing.assert_close(
        torch.tensor(result["schedule"]["sigmas"]), torch.tensor(schedule["sigmas"]), rtol=0, atol=1e-6
    )
    for index, timestep in enumerate(result["schedule"]["timesteps"]):
        saved = torch.load(
            capture / f"step_{index:03d}/transformer/input/timestep.pt", map_location="cpu", weights_only=True
        )
        assert timestep == saved.item()
