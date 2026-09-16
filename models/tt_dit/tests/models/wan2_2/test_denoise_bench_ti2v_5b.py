# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level profile of the TI2V-5B denoise loop.

The same treatment that found the VAE's 78%-in-one-module problem, applied to the
section that is now ~91% of the run (12.18 s of a 13.39 s 720p T2V generation). The
denoise runs traced in production, so it is profiled here on the **untraced** path
(``pipeline(..., traced=False)``): tracing removes host dispatch but not device work,
so the per-op *device* ranking from sync mode is the same either way, and the
dispatch-mode number tells us how much host overhead tracing is already hiding.

Callsites attribute the work: ``transformer_wan.py`` / ``attention_wan.py`` /
``linear.py`` / ``normalization.py`` / ``manager.py`` lines are denoise;
``vae_wan2_1.py`` is the decode; the text encoder has its own file. Keep steps low --
every op is synchronised in sync mode.

    WAN5B_DENOISE_STEPS=2 pytest \
      models/tt_dit/tests/models/wan2_2/test_denoise_bench_ti2v_5b.py -sv --timeout=0
"""

import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests dir is not a package
from _ttnn_host_profiler import profile_ttnn_ops  # noqa: E402

_PROMPT = "A neon-lit street at night in the rain, reflections on wet asphalt, " "cinematic, highly detailed, 24fps"


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_denoise_profile_ti2v_5b(mesh_device, mesh_shape, topology):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B manual bring-up is targeting BH Galaxy first")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height = int(os.environ.get("WAN5B_DENOISE_HEIGHT", 704))
    width = int(os.environ.get("WAN5B_DENOISE_WIDTH", 1280))
    num_frames = int(os.environ.get("WAN5B_DENOISE_FRAMES", 81))
    steps = int(os.environ.get("WAN5B_DENOISE_STEPS", 2))

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=height,
        width=width,
        num_frames=num_frames,
        run_warmup=False,
    )

    def _run():
        with torch.no_grad():
            return pipeline(
                prompts=[_PROMPT],
                num_inference_steps=steps,
                seed=42,
                output_type="uint8",
                traced=False,
            )

    logger.info(f"denoise profile: {height}x{width}, {num_frames}f, {steps} untraced steps")

    t0 = time.perf_counter()
    _run()  # compile programs for these shapes before measuring anything
    logger.info(f"DENOISE_WARMUP: {time.perf_counter() - t0:.2f}s")

    with profile_ttnn_ops(mode="dispatch") as p:
        _run()
    logger.info("\n" + p.result.report(f"untraced {steps}-step denoise {width}x{height}"))

    with profile_ttnn_ops(mode="sync", device=mesh_device) as p2:
        _run()
    logger.info("\n" + p2.result.report(f"untraced {steps}-step denoise {width}x{height}"))

    assert p2.result.count > 0, "profiler recorded no ops"
