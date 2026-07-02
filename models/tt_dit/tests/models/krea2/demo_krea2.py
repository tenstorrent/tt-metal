# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end KREA-2 (Krea 2 Turbo) tt_dit demo / driver.

Constructs the KREA-2 pipeline on a 2x4 mesh and generates one image. This is the script
the main thread runs on device (author-side did NOT run any device code).

Run (2x4 Wormhole/Blackhole):
    pytest models/tt_dit/tests/models/krea2/demo_krea2.py

or directly (build a device yourself and call `run_demo`).

Environment:
    KREA2_PROMPT   optional prompt override (default: a fox in the snow)
    KREA2_STEPS    optional step count (default: 8, the Turbo setting)
    KREA2_SEED     optional seed (default: 0)
    KREA2_OUT      optional output PNG path (default: krea2_demo.png)
    HF_HOME / HF_HUB_CACHE   HF cache root (defaults to /localdev/vsuresh/hf_cache)
"""

import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.krea2.pipeline_krea2 import Krea2Pipeline


def run_demo(
    mesh_device: ttnn.MeshDevice,
    *,
    prompt: str,
    num_inference_steps: int,
    seed: int,
    height: int,
    width: int,
    out_path: str,
) -> None:
    logger.info("creating KREA-2 pipeline ({}x{} mesh {})...", height, width, tuple(mesh_device.shape))
    pipeline = Krea2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        width=width,
        height=height,
        use_torch_vae_decoder=bool(os.environ.get("KREA2_TORCH_VAE")),
    )

    logger.info("generating for prompt: {!r}", prompt)
    images = pipeline.run(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        height=height,
        width=width,
        output_type="pil",
    )

    images[0].save(out_path)
    logger.info("saved image to {}", out_path)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 47000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024)])
def test_krea2_demo(*, mesh_device: ttnn.MeshDevice, height: int, width: int) -> None:
    prompt = os.environ.get("KREA2_PROMPT", "a fox in the snow")
    steps = int(os.environ.get("KREA2_STEPS", "8"))
    seed = int(os.environ.get("KREA2_SEED", "0"))
    out_path = os.environ.get("KREA2_OUT", "krea2_demo.png")

    run_demo(
        mesh_device,
        prompt=prompt,
        num_inference_steps=steps,
        seed=seed,
        height=height,
        width=width,
        out_path=out_path,
    )
