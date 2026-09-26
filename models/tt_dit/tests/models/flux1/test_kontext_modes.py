# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate the FLUX.1-Kontext-dev pipeline extensions in one device session:

  A) edit @ 1024x1024        (regression — reference image + instruction)
  B) generate @ 1024x1024    (image=None -> plain text-to-image)
  C) edit @ 1104x944         (per-request resolution, a Kontext preferred bucket)

The pipeline (12B) is created once and reused across the three calls, so weights
load a single time. Each new (mode, resolution) shape triggers a first-call
kernel recompile (untraced), which is expected.
"""


import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from ....pipelines.flux1.pipeline_flux1_kontext import Flux1KontextPipeline, Flux1KontextPipelineConfig


def _gradient(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    px = img.load()
    for x in range(width):
        for y in range(height):
            px[x, y] = (int(255 * x / width), int(255 * y / height), 128)
    return img


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 50000000,
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp", "encoder_tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
    [
        pytest.param((1, 2), (1, 0), (2, 1), (2, 1), (2, 1), ttnn.Topology.Linear, 2, "1x2sp0tp1", id="1x2sp0tp1"),
    ],
    indirect=["mesh_device"],
)
def test_flux1_kontext_modes(
    *,
    mesh_device: ttnn.MeshDevice,
    sp,
    tp,
    encoder_tp,
    vae_tp,
    topology,
    num_links,
    mesh_test_id,
    model_location_generator,
) -> None:
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=sp, tp=tp)

    pipeline = Flux1KontextPipeline(
        device=mesh_device,
        config=Flux1KontextPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=parallel_config,
            encoder_parallel_config=EncoderParallelConfig.from_tuple(encoder_tp),
            vae_parallel_config=VAEParallelConfig.from_tuple(vae_tp),
            num_links=num_links,
            topology=topology,
            width=1024,
            height=1024,
            checkpoint_name=model_location_generator("black-forest-labs/FLUX.1-Kontext-dev"),
        ),
    )

    def run(**kw):
        imgs = pipeline(num_inference_steps=20, guidance_scale=3.5, seed=0, traced=False, **kw)
        assert len(imgs) == 1
        return imgs[0]

    # A) edit @ 1024 (regression)
    a = run(image=_gradient(1024, 1024), prompts=["Add a small red hat to the subject"])
    a.save("kontext_mode_edit_1024.png")
    assert a.size == (1024, 1024)
    logger.info("A edit@1024 OK")

    # B) text-to-image @ 1024 (no reference image)
    b = run(image=None, prompts=["A photo of a red vintage car on a mountain road, golden hour"])
    b.save("kontext_mode_generate_1024.png")
    assert b.size == (1024, 1024)
    logger.info("B generate@1024 OK")

    # C) edit @ 1104x944 (per-request preferred resolution)
    c = run(image=_gradient(1104, 944), width=1104, height=944, prompts=["Make it a watercolor painting"])
    c.save("kontext_mode_edit_1104x944.png")
    assert c.size == (1104, 944)
    logger.info("C edit@1104x944 OK")

    logger.info("ALL KONTEXT MODES PASSED")
