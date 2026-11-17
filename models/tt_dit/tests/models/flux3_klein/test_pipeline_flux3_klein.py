# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import PRECISION_BF16, PRECISION_FAST
from ....pipelines.flux3_klein.pipeline_flux3_klein import SHAPES, Flux3KleinPipeline, Flux3KleinRequest
from ....utils.test import ring_params_8k

PROMPT = "A man in a red jacket walks a golden retriever along a snowy forest path at sunrise, photorealistic"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8), (4, 32)], ids=["galaxy", "quad"], indirect=True)
@pytest.mark.parametrize("model", ["distilled"])
def test_pipeline(*, mesh_device: ttnn.MeshDevice, model: str) -> None:
    """On a quad (4 x 32, one tt-run rank per Galaxy) rank 0 holds the images; the other ranks get None."""
    pipeline = Flux3KleinPipeline.create(mesh_device, models=(model,))
    pipeline.warmup()

    reference_path = os.environ.get("FLUX3_KLEIN_REFERENCE_IMAGE")
    reference = Image.open(reference_path) if reference_path else Image.new("RGB", (1024, 1024), (90, 120, 160))
    for width, height in SHAPES:
        for images in ((), (reference,)):
            request = Flux3KleinRequest(PROMPT, model=model, width=width, height=height, seed=0, images=images)
            t0 = time.time()
            result = pipeline.generate(request)
            total = time.time() - t0
            filename = f"flux3_klein_{model}_{width}x{height}_refs{len(images)}.png"
            logger.info(f"{filename}: {total:.2f} s end to end, stages {result.timings}")
            if result.image is not None:
                result.image.save(filename)
                assert result.image.size == (width, height)
    pipeline.close()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["galaxy"], indirect=True)
def test_pipeline_fast(*, mesh_device: ttnn.MeshDevice) -> None:
    """The fast mode renders the same request deterministically, at least 25 dB PSNR against the bf16 image."""
    request = Flux3KleinRequest(PROMPT, model="distilled", width=1024, height=1024, seed=0)
    images = {}
    for precision in (PRECISION_BF16, PRECISION_FAST):
        pipeline = Flux3KleinPipeline.create(mesh_device, precision=precision)
        runs = 2 if precision is PRECISION_FAST else 1  # bf16 once, the fast mode twice for determinism
        images[precision.name] = [np.asarray(pipeline.generate(request).image, dtype=np.float64) for _ in range(runs)]
        pipeline.close()
    (bf16,), (fast, fast_again) = images.values()
    assert fast.shape == (1024, 1024, 3) and fast.std() > 10
    assert np.array_equal(fast, fast_again)
    psnr = 10 * np.log10(255**2 / np.mean((fast - bf16) ** 2))
    logger.info(f"fast mode vs bf16: {psnr:.2f} dB")
    assert psnr >= 25
