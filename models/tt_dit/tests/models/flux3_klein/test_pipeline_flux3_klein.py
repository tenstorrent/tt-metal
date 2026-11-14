# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.dbcache_flux3_klein import flux3_klein_dbcache_config
from ....pipelines.flux3_klein.pipeline_flux3_klein import SIZES, Flux3KleinPipeline, Flux3KleinRequest
from ....utils.test import ring_params_8k

PROMPT = "A man in a red jacket walks a golden retriever along a snowy forest path at sunrise, photorealistic"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("model", ["distilled"])
@pytest.mark.parametrize("dbcache", [False, True], ids=["no_cache", "dbcache"])
def test_pipeline(*, mesh_device: ttnn.MeshDevice, model: str, dbcache: bool) -> None:
    pipeline = Flux3KleinPipeline.create(mesh_device, models=(model,))
    pipeline.warmup()

    reference = Image.new("RGB", (1024, 1024), (90, 120, 160))
    for size in SIZES:
        for images in ((), (reference,)):
            cache = flux3_klein_dbcache_config(len(images)) if dbcache else None
            request = Flux3KleinRequest(
                PROMPT, model=model, width=size, height=size, seed=0, images=images, cache=cache
            )
            t0 = time.time()
            result = pipeline.generate(request)
            total = time.time() - t0
            filename = f"flux3_klein_{model}_{size}_refs{len(images)}{'_dbcache' if dbcache else ''}.png"
            result.image.save(filename)
            logger.info(f"{filename}: {total:.2f} s end to end, {result.cached_steps} cached steps, {result.timings}")
            assert result.image.size == (size, size)
            assert float(np.asarray(result.image, dtype=np.float32).std()) >= 10, "flat image"
            assert (result.cached_steps > 0) == dbcache
            assert pipeline.generate(request).image.tobytes() == result.image.tobytes()
    pipeline.close()
