# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os
import statistics
import time

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.pipelines.flux3_klein.pipeline_flux3_klein import Flux3KleinPipeline, Flux3KleinRequest
from models.tt_dit.utils.test import ring_params_8k

PROMPT = "A man in a red jacket walks a golden retriever along a snowy forest path at sunrise, photorealistic"
SEED = 123
REPEATS = 3
REFERENCES = {"none": None, "1mp": (1024, 1024), "4mp": (2048, 2048)}
RESULTS_FILE = os.environ.get("FLUX3_RESULTS", "flux3_klein_results.jsonl")

CONFIGS = {
    "full_sp8_tp4_ring": (None, 0, 1, ttnn.Topology.Ring),
    "tp4_sp2_linear": ((4, 2), 0, 1, ttnn.Topology.Linear),
    "tp2_sp4_linear": ((4, 2), 1, 0, ttnn.Topology.Linear),
    "tp4_sp2_ring": ((4, 2), 0, 1, ttnn.Topology.Ring),
    "tp2_sp4_ring": ((4, 2), 1, 0, ttnn.Topology.Ring),
}


def _record(**fields) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(fields) + "\n")


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("size", [1024, 2048, 4096])
def test_submesh_pipeline(*, mesh_device: ttnn.MeshDevice, config: str, size: int) -> None:
    submesh_shape, tp_axis, sp_axis, topology = CONFIGS[config]
    device = mesh_device if submesh_shape is None else mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    shape = tuple(device.shape)
    logger.info(f"{config} {size}: mesh {shape}, tp={shape[tp_axis]}, sp={shape[sp_axis]}, topology={topology}")

    try:
        pipeline = Flux3KleinPipeline.create(device, tp_axis=tp_axis, sp_axis=sp_axis, topology=topology)
        t0 = time.time()
        pipeline.warmup((size,))
        logger.info(f"{config} {size}: warmup in {time.time() - t0:.1f} s")
    except Exception as e:
        _record(config=config, size=size, reference="all", status=f"warmup failed: {type(e).__name__}: {e}"[:300])
        raise

    for reference, ref_size in REFERENCES.items():
        images = () if ref_size is None else (Image.new("RGB", ref_size, (90, 120, 160)),)
        request = Flux3KleinRequest(PROMPT, width=size, height=size, seed=SEED, images=images)
        try:
            totals, timings = [], []
            for _ in range(REPEATS):
                t0 = time.time()
                result = pipeline.generate(request)
                totals.append(time.time() - t0)
                timings.append(result.timings)
        except Exception as e:
            _record(config=config, size=size, reference=reference, status=f"failed: {type(e).__name__}: {e}"[:300])
            raise
        result.image.save(f"flux3_klein_{config}_{size}_ref_{reference}.png")
        median = statistics.median(totals)
        stages = {k: round(statistics.median(t[k] for t in timings), 3) for k in timings[0]}
        flat = float(np.asarray(result.image, dtype=np.float32).std()) < 10
        _record(
            config=config, size=size, reference=reference, status="flat image" if flat else "ok",
            median_s=round(median, 3), runs_s=[round(t, 3) for t in totals], stages=stages,
        )
        logger.info(f"RESULT {config} {size} ref_{reference}: median {median:.2f} s of {totals}, stages {stages}")
        assert result.image.size == (size, size)
        assert not flat, "flat image"
    pipeline.close()
