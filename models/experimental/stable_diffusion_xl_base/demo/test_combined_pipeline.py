import pytest
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_combined_pipeline import TtSDXLCombinedPipeline
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import TtSDXLImg2ImgPipelineConfig
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipelineConfig


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 33000, "trace_region_size": 58378240}],
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

    # Create combined pipeline using factory method
    # This automatically creates base and refiner pipelines with a shared scheduler
    combined = TtSDXLCombinedPipeline.create(
        ttnn_device=mesh_device,
        torch_base_pipeline=base,
        torch_refiner_pipeline=refiner,
        base_config=TtSDXLPipelineConfig(
            num_inference_steps=10,
            guidance_scale=7.5,
            is_galaxy=False,
            use_cfg_parallel=False,
            encoders_on_device=True,
            vae_on_device=False,
        ),
        refiner_config=TtSDXLImg2ImgPipelineConfig(
            num_inference_steps=10,
            guidance_scale=7.5,
            is_galaxy=False,
            use_cfg_parallel=False,
            vae_on_device=True,
            encoders_on_device=False,
            strength=0.3,
            aesthetic_score=6.0,
            negative_aesthetic_score=2.5,
        ),
        denoising_split=0.8,  # Base does 80%, refiner does 20%
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
        num_inference_steps=10,
        guidance_scale=7.5,
        crop_coords_top_left=(0, 0),
        guidance_rescale=0.0,
    )
