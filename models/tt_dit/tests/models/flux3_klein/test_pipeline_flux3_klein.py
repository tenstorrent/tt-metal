# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.dbcache_flux3_klein import flux3_klein_dbcache_config
from ....pipelines.flux3_klein.pipeline_flux3_klein import SIZES, Flux3KleinPipeline, Flux3KleinRequest, _plan_key
from ....utils.test import ring_params_8k

PROMPT = "A man in a red jacket walks a golden retriever along a snowy forest path at sunrise, photorealistic"


def test_free_other_plans_keeps_the_request_layout():
    freed = []
    ids = {"image": torch.zeros(1, 4)}

    def key(image_rows, cond_rows):
        return _plan_key("distilled", 0, 4, {"image": image_rows, "image_cond": cond_rows, "text": 512}, 7, ids)

    def plan(name):
        return types.SimpleNamespace(deallocate=lambda: freed.append(name))

    pipeline = object.__new__(Flux3KleinPipeline)
    pipeline._plan_cache = OrderedDict(
        [(key(65536, 1024), plan("same")), (key(4096, 1024), plan("image")), (key(65536, 256), plan("cond"))]
    )
    pipeline._free_other_plans(65536, 1024)
    assert list(pipeline._plan_cache) == [key(65536, 1024)] and sorted(freed) == ["cond", "image"]


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
