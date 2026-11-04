# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.pipeline_flux3_klein import SIZES, Flux3KleinPipeline, Flux3KleinRequest
from ....utils.test import ring_params_8k

PROMPT = "A man in a red jacket walks a golden retriever along a snowy forest path at sunrise, photorealistic"


@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("model", ["distilled"])
def test_pipeline(*, mesh_device: ttnn.MeshDevice, model: str) -> None:
    pipeline = Flux3KleinPipeline.create(mesh_device, models=(model,), upsample=False)
    pipeline.warmup()

    reference_path = os.environ.get("FLUX3_KLEIN_REFERENCE_IMAGE")
    reference = Image.open(reference_path) if reference_path else Image.new("RGB", (1024, 1024), (90, 120, 160))
    for size in SIZES:
        for images in ((), (reference,)):
            request = Flux3KleinRequest(PROMPT, model=model, width=size, height=size, seed=0, images=images)
            t0 = time.time()
            result = pipeline.generate(request)
            total = time.time() - t0
            filename = f"flux3_klein_{model}_{size}_refs{len(images)}.png"
            result.image.save(filename)
            logger.info(f"{filename}: {total:.2f} s end to end, stages {result.timings}")
            assert result.image.size == (size, size)
    pipeline.close()
