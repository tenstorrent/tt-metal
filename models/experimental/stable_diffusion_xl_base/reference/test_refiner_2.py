from diffusers import DiffusionPipeline
import torch

torch.manual_seed(0)

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True
)
base.to("cpu")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float32,
    use_safetensors=True,
)
refiner.to("cpu")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "An astronaut riding a green horse"

# run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
# ).images
image = torch.randn(1, 4, 128, 128)  # dummy latent for testing
for k in refiner.unet.state_dict().keys():
    print(k)
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
image.save("refined_output.jpg")
