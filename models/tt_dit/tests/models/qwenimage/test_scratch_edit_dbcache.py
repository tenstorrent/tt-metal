# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Throwaway: DBCache on every edit size and image count, against the same edit without it.
import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image, ImageOps

import ttnn
from models.tt_dit.pipelines.events import SectionEnd, SectionStart
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    DEFAULT_DBCACHE,
    QwenImagePipeline,
    QwenImagePipelineConfig,
)
from models.tt_dit.utils.dbcache import DBCacheConfig

from .device_params import EDIT_MESH_DEVICE_PARAMS
from .test_pipeline_qwenimage import DOG_IMAGE_PATH

DEVICE_PARAMS = [
    pytest.param(
        p.values[0], {**p.values[1], "trace_region_size": int(p.values[1]["trace_region_size"] * 1.1)}, id=p.id
    )
    for p in EDIT_MESH_DEVICE_PARAMS
]

SINGLE_PROMPT = "Make the dog wear a red knitted scarf."
MULTI_PROMPT = "Put the dogs from all the images together on one sofa."


@pytest.mark.parametrize(("mesh_device", "device_params"), DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "checkpoint_name",
    [
        pytest.param("Qwen/Qwen-Image-Edit", id="edit"),
        pytest.param("Qwen/Qwen-Image-Edit-2509", id="2509"),
        pytest.param("Qwen/Qwen-Image-Edit-2511", id="2511"),
    ],
)
def test_scratch_edit_dbcache(*, mesh_device: ttnn.MeshDevice, checkpoint_name: str) -> None:
    config = QwenImagePipelineConfig.default(
        mesh_shape=mesh_device.shape, checkpoint_name=checkpoint_name, dbcache=DEFAULT_DBCACHE
    )
    pipeline = QwenImagePipeline(device=mesh_device, config=config)
    assert pipeline._dbcache is not None
    dog = Image.open(DOG_IMAGE_PATH).convert("RGB")
    sizes = config.resolutions

    # Each size with one input image of its aspect ratio, then 2 and 3 images of mixed ones; the
    # output takes the last image's size.
    cases = [[size] for size in sizes]
    if config.max_images >= 2:
        cases.append([sizes[1], sizes[0]])
    if config.max_images >= 3:
        cases.append([sizes[2], sizes[4], sizes[0]])

    tag = checkpoint_name.rsplit("/", 1)[-1].lower()
    out_dir = Path(f"qwenimage_edit_dbcache/{tag}_{mesh_device.shape[0]}x{mesh_device.shape[1]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        images = [ImageOps.fit(dog, size) for size in case]
        prompt = SINGLE_PROMPT if len(images) == 1 else MULTI_PROMPT
        outputs, times = {}, {}
        for name, dbcache in [("full", DBCacheConfig(residual_diff_threshold=0)), ("dbcache", None)]:
            starts: dict[str, float] = {}

            def on_event(event: object) -> None:
                if isinstance(event, SectionStart) and event.name == "denoising":
                    starts["denoising"] = time.perf_counter()
                elif isinstance(event, SectionEnd) and event.name == "denoising":
                    times[name] = time.perf_counter() - starts["denoising"]

            (edited,) = pipeline(
                prompts=[prompt],
                num_inference_steps=50,
                seed=0,
                image=images,
                dbcache=dbcache,
                on_event=on_event,
            )
            label = f"{edited.width}x{edited.height}_{len(images)}img"
            edited.save(out_dir / f"{label}_{name}.png")
            outputs[name] = np.asarray(edited, dtype=np.float64)

        full, cached = outputs["full"], outputs["dbcache"]
        mse = np.mean((full - cached) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
        pcc = np.corrcoef(full.ravel(), cached.ravel())[0, 1]
        skipped = len(pipeline._dbcache.cached_steps)
        rows.append((label, skipped, times["full"] / times["dbcache"], psnr, pcc))

    logger.info(f"{tag} on {mesh_device.shape}: size, skipped, denoising speedup, PSNR, PCC")
    for label, skipped, speedup, psnr, pcc in rows:
        logger.info(f"  {label}: {skipped}/50, {speedup:.2f}x, {psnr:.1f} dB, {pcc:.4f}")

    for label, _, _, psnr, _ in rows:
        assert psnr >= 25, f"{label}: {psnr:.1f} dB"
