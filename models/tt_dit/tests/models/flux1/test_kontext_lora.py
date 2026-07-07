# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate FLUX.1 LoRA fuse-in-load on the Kontext pipeline.

Set LORA_PATH (a FLUX.1 LoRA .safetensors) and optionally LORA_SCALE. The LoRA
is fused into the transformer weights at load time (Flux1Checkpoint), then a
text-to-image generation is run so the style change is visible. Run with an
empty LORA_PATH for the no-LoRA baseline (same prompt/seed) to A/B compare.
"""

import os

import pytest
from loguru import logger

import ttnn

from ....parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from ....pipelines.flux1.pipeline_flux1_kontext import Flux1KontextPipeline, Flux1KontextPipelineConfig


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
    [pytest.param((1, 2), (1, 0), (2, 1), (2, 1), (2, 1), ttnn.Topology.Linear, 2, "1x2sp0tp1", id="1x2sp0tp1")],
    indirect=["mesh_device"],
)
def test_flux1_kontext_lora(
    *, mesh_device, sp, tp, encoder_tp, vae_tp, topology, num_links, mesh_test_id, model_location_generator
) -> None:
    lora_path = os.environ.get("LORA_PATH") or None
    lora_scale = float(os.environ.get("LORA_SCALE", "1.0"))
    tag = os.environ.get("LORA_TAG", "lora" if lora_path else "base")
    logger.info(f"LoRA test: lora_path={lora_path!r} scale={lora_scale} tag={tag}")

    pipeline = Flux1KontextPipeline(
        device=mesh_device,
        config=Flux1KontextPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=sp, tp=tp),
            encoder_parallel_config=EncoderParallelConfig.from_tuple(encoder_tp),
            vae_parallel_config=VAEParallelConfig.from_tuple(vae_tp),
            num_links=num_links,
            topology=topology,
            width=1024,
            height=1024,
            checkpoint_name=model_location_generator("black-forest-labs/FLUX.1-Kontext-dev"),
            lora_path=lora_path,
            lora_scale=lora_scale,
        ),
    )

    prompt = os.environ.get("PROMPT", "portrait of a young woman with long hair, upper body, simple background")
    imgs = pipeline(prompts=[prompt], num_inference_steps=24, guidance_scale=3.5, seed=0, traced=False)
    assert len(imgs) == 1
    out = f"kontext_lora_{tag}.png"
    imgs[0].save(out)
    logger.info(f"saved {out}  (LoRA fuse-in-load {'ON' if lora_path else 'OFF'})")
    logger.info("KONTEXT_LORA_OK")
