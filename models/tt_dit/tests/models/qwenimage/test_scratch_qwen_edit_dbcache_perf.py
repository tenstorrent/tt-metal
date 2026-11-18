# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: Qwen-Image-Edit-2511 with DBCache and 1 to 3 input images, built as tt-media-server builds it
(server device params, 40 steps, guidance 4.0, 1024x1024), to compare a Galaxy with the server's
release run."""

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    DEFAULT_DBCACHE,
    QwenImagePipeline,
    QwenImagePipelineConfig,
)
from models.tt_dit.utils.test import line_params_req_exact_devices

from .test_performance_qwenimage import EDIT_PROMPT, crop_to_aspect_ratio
from .test_pipeline_qwenimage import DOG_IMAGE_PATH

RUNS = 3
SECTIONS = ("encoder", "denoising", "vae")


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param(
            (4, 8),
            {**line_params_req_exact_devices, "l1_small_size": 32768, "trace_region_size": 869_000_000},
            id="4x8",
        )
    ],
    indirect=True,
)
def test_qwen_edit_dbcache_perf(*, mesh_device: ttnn.MeshDevice, is_ci_env: bool, monkeypatch) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    config = QwenImagePipelineConfig.default(
        mesh_shape=mesh_device.shape, checkpoint_name="Qwen/Qwen-Image-Edit-2511", dbcache=DEFAULT_DBCACHE
    )
    pipeline = QwenImagePipeline(device=mesh_device, config=config)
    dog = crop_to_aspect_ratio(Image.open(DOG_IMAGE_PATH).convert("RGB"), (1, 1))

    rows = []
    for n in (1, 2, 3):
        for run in range(RUNS):
            profiler = BenchmarkProfiler()
            with profiler("run", iteration=0):
                pipeline(
                    prompts=[EDIT_PROMPT],
                    image=[dog] * n,
                    seed=0,
                    width=1024,
                    height=1024,
                    num_inference_steps=40,
                    cfg_scale=4.0,
                    on_event=profiler_event_callback(profiler, 0),
                )
            t = {s: profiler.get_duration(s, 0) for s in SECTIONS if profiler.contains_step(s, 0)}
            skipped = len(pipeline._dbcache.cached_steps) if pipeline._dbcache is not None else "-"
            row = (
                f"QWENGLX {n} inputs | run {run} | total {profiler.get_duration('run', 0):6.2f} s | "
                + " | ".join(f"{s} {t[s]:6.2f} s" for s in t)
                + f" | skipped {skipped}"
            )
            logger.info(row)
            rows.append(row)
    print("\n".join(rows))
