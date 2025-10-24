import pytest
import torch
from diffusers import DiffusionPipeline

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_combined_pipeline import (
    TtSDXLCombinedPipeline,
    TtSDXLCombinedPipelineConfig,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 23000, "trace_region_size": 27082752}],
    indirect=True,
)
def test_refiner(mesh_device):
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    tt_base = TtSDXLPipeline(
        ttnn_device=mesh_device,
        torch_pipeline=base,
        pipeline_config=TtSDXLPipelineConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            is_galaxy=False,
            skip_vae_decode=True,
            use_cfg_parallel=False,
            encoders_on_device=True,
        ),
    )

    tt_refiner = TtSDXLPipeline(
        ttnn_device=mesh_device,
        torch_pipeline=refiner,
        pipeline_config=TtSDXLPipelineConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            is_galaxy=False,
            skip_vae_decode=False,
            use_cfg_parallel=False,
            vae_on_device=True,
        ),
    )

    config = TtSDXLCombinedPipelineConfig(
        base_config=tt_base.pipeline_config,
        refiner_config=tt_refiner.pipeline_config,
        denoising_split=0.8,
        use_refiner=True,
    )

    combined = TtSDXLCombinedPipeline(
        ttnn_device=mesh_device,
        tt_base_pipeline=tt_base,
        tt_refiner_pipeline=tt_refiner,
        config=config,
    )

    combined.generate(
        prompts=["A beautiful sunset over a calm ocean"],
        negative_prompts=[""],
        prompt_2=None,
        negative_prompt_2=None,
        start_latent_seed=None,
        fixed_seed_for_batch=False,
        timesteps=None,
        sigmas=None,
    )
