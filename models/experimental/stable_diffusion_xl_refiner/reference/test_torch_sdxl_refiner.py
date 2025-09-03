import pytest
import torch
from diffusers import DiffusionPipeline

# , "A serene mountain landscape"


@pytest.mark.parametrize("prompt", ["A futuristic city at sunset"])
@pytest.mark.parametrize("num_inference_steps", [20])
def test_sdxl_refiner_pipeline(is_ci_env, prompt, num_inference_steps):
    if is_ci_env:
        pytest.skip("Skipping test in CI environment")

    torch.manual_seed(42)

    # Load the base pipeline
    # base_pipe = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.bfloat16,
    #     use_safetensors=True
    # )

    # Load the refiner pipeline
    refiner_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.bfloat16, use_safetensors=True
    )

    # Generate base image
    # base_image = base_pipe(
    #     prompt=prompt,
    #     num_inference_steps=num_inference_steps,
    #     denoising_end=0.8,
    #     output_type="latent"
    # ).images

    # Create random latent tensor as input for refiner
    # SDXL latent dimensions: (batch_size, channels=4, height=128, width=128)
    random_latent = torch.randn(1, 4, 128, 128, dtype=torch.bfloat16)

    # Refine the image
    refined_image = refiner_pipe(
        prompt=prompt, image=random_latent, num_inference_steps=num_inference_steps, denoising_start=0.0
    ).images

    # Assertions
    assert refined_image is not None
    assert len(refined_image) > 0
    refined_image[0].save(f"refined_test_output_{hash(prompt)}.png")
