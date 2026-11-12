# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full checkpoint reproduction against independently generated Diffusers outputs.

The reference files are required: missing inputs fail rather than silently skipping CI.
Use QWEN_IMAGE_GOLDENS and QWEN_IMAGE_RESULTS to keep artifacts outside the source tree.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR, HF_REVISION
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline


@pytest.fixture(scope="module")
def pipeline():
    device = open_device()
    pipe = None
    try:
        pipe = QwenImage21Pipeline(device)
        yield pipe
    finally:
        try:
            if pipe is not None:
                pipe.release_traces()
        finally:
            close_device(device)


def correlation(actual, expected):
    actual = torch.as_tensor(actual).float().flatten()
    expected = torch.as_tensor(expected).float().flatten()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    return torch.corrcoef(torch.stack((actual, expected)))[0, 1].item()


def rgb_over_white(image):
    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return np.asarray(rgb).copy()


@pytest.mark.parametrize("case,latent_threshold", [("text", 0.99), ("edit", 0.98), ("edit2", 0.98)])
def test_full_pipeline_against_diffusers(pipeline, case, latent_threshold, record_property):
    golden = Path(GOLDENS_DIR) / ("" if case == "text" else case)
    meta = json.loads((golden / "meta.json").read_text())
    assert meta["revision"] == HF_REVISION
    assert meta["steps"] == 40 and meta["seed"] == 42
    images = None if case == "text" else [Image.open(path).convert("RGBA") for path in meta["image"]]
    denoise_name = "denoise.pt" if case == "text" else "denoise_edit.pt"
    image_name = "image.png" if case == "text" else "image_edit.png"
    reference = torch.load(golden / denoise_name, map_location="cpu", weights_only=False)
    expected = reference["latents_after_step"][-1][2]
    expected_image = Image.open(golden / image_name)
    results = Path(os.environ.get("QWEN_IMAGE_RESULTS", "generated/qwen_image_2_1/results"))
    results.mkdir(parents=True, exist_ok=True)
    timings = []
    previous = None
    for repetition in range(4):
        rgb, rgba, latents, timing = pipeline.generate(meta["prompt"], seed=42, num_steps=40, images=images)
        assert rgb.size == expected_image.size
        assert torch.isfinite(latents).all()
        if previous is not None:
            assert torch.equal(latents, previous), "trace replay changed the repeated request"
        previous = latents.clone()
        timings.append(timing.as_dict())
    latent_pcc = correlation(latents, expected)
    pixel_pcc = correlation(rgb_over_white(rgb), rgb_over_white(expected_image))
    record = {
        "case": case,
        "checkpoint_revision": HF_REVISION,
        "reference": meta,
        "warmup_requests": 1,
        "measured_requests": 3,
        "timings": timings,
        "latent_pcc": latent_pcc,
        "pixel_pcc": pixel_pcc,
    }
    (results / f"{case}.json").write_text(json.dumps(record, indent=2))
    rgb.save(results / f"{case}.png")
    rgba.save(results / f"{case}-rgba.png")
    torch.save(latents, results / f"{case}-latents.pt")
    record_property("latent_pcc", latent_pcc)
    record_property("pixel_pcc", pixel_pcc)
    print(json.dumps(record), flush=True)
    assert latent_pcc >= latent_threshold
    assert pixel_pcc >= 0.90
