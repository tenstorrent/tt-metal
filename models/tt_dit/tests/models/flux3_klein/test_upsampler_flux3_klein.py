# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.pipeline_flux3_klein import SIZES, Flux3KleinPipeline, Flux3KleinRequest
from ....pipelines.flux3_klein.upsampler_flux3_klein import DEVICE_PARAMS, SHARED_MESH_PARAMS, Flux3KleinUpsampler
from ....utils.test import ring_params_8k

REQUESTS = [
    ("A man walking his dog in the snow", None),
    ("Make it a night scene with street lights", Image.new("RGB", (1600, 900), (120, 140, 90))),
]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_upsampler(*, mesh_device: ttnn.MeshDevice) -> None:
    upsampler = Flux3KleinUpsampler(mesh_device)
    for prompt, image in REQUESTS:
        t0 = time.time()
        caption = upsampler(prompt, image)
        logger.info(f"{prompt!r} -> {len(caption)} characters in {time.time() - t0:.1f} s: {caption}")
        assert caption and caption != prompt
    upsampler.close()


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(
    "device_params", [{**ring_params_8k, "l1_small_size": 65536, **SHARED_MESH_PARAMS}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_pipeline_upsampled(*, mesh_device: ttnn.MeshDevice) -> None:
    pipeline = Flux3KleinPipeline.create(mesh_device, upsample=True)
    pipeline.warmup()
    for size in SIZES:
        for prompt, image in REQUESTS:
            images = () if image is None else (image,)
            request = Flux3KleinRequest(prompt, width=size, height=size, seed=0, images=images)
            first, second = pipeline.generate(request), pipeline.generate(request)
            filename = f"flux3_klein_upsampled_{size}_refs{len(images)}.png"
            first.image.save(filename)
            logger.info(f"{filename}: stages {first.timings}, caption: {first.prompt}")
            assert first.prompt != prompt and first.prompt == second.prompt
            assert first.image.tobytes() == second.image.tobytes()
    pipeline.close()
