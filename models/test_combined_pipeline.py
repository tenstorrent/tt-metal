import pytest
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_combined_pipeline import (
    TtSDXLCombinedPipeline,
    TtSDXLCombinedPipelineConfig,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLImg2ImgPipeline,
    TtSDXLImg2ImgPipelineConfig,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 23000, "trace_region_size": 27082752}],
    indirect=True,
)
def test_refiner(mesh_device):
    # Load base and refiner torch pipelines
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    # Create TT base pipeline (text-to-image)
    tt_base = TtSDXLPipeline(
        ttnn_device=mesh_device,
        torch_pipeline=base,
        pipeline_config=TtSDXLPipelineConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            is_galaxy=False,
            use_cfg_parallel=False,
            encoders_on_device=True,
        ),
    )

    # Create TT refiner pipeline (image-to-image)
    tt_refiner = TtSDXLImg2ImgPipeline(
        ttnn_device=mesh_device,
        torch_pipeline=refiner,
        pipeline_config=TtSDXLImg2ImgPipelineConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            is_galaxy=False,
            use_cfg_parallel=False,
            vae_on_device=True,
            strength=0.3,
            aesthetic_score=6.0,
            negative_aesthetic_score=2.5,
        ),
    )

    # Create combined pipeline config (denoising_split < 1.0 means refiner is enabled)
    config = TtSDXLCombinedPipelineConfig(
        base_config=tt_base.pipeline_config,
        refiner_config=tt_refiner.pipeline_config,
        denoising_split=1.0,  # Base does 80%, refiner does 20%
    )

    # Create combined pipeline
    combined = TtSDXLCombinedPipeline(
        ttnn_device=mesh_device,
        tt_base_pipeline=tt_base,
        tt_refiner_pipeline=tt_refiner,
        config=config,
    )

    # Generate images
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
