import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

torch.manual_seed(0)
# -----------------------------
# CONFIGURATION
# -----------------------------
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

# If running offline, replace with local paths, e.g.:
# base_model_id = "/path/to/sdxl-base"
# refiner_model_id = "/path/to/sdxl-refiner"

device = "cpu"
prompt = "An astronaut riding a green horse"

# -----------------------------
# STAGE 1: BASE IMAGE GENERATION
# -----------------------------
print("Loading SDXL base pipeline...")
pipe_base = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
pipe_base = pipe_base.to(device)

print("Generating base image...")
image = pipe_base(prompt=prompt, num_inference_steps=32).images[0]
# image.save("base_output.jpg")
print("✅ Generated base image.")

# -----------------------------
# STAGE 2: REFINER
# -----------------------------
print("Loading SDXL refiner pipeline...")
pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_model_id, torch_dtype=torch.float32)
pipe_refiner = pipe_refiner.to(device)

print("Refining the image...")
refined_image = pipe_refiner(
    prompt=prompt,
    image=image,
    num_inference_steps=8,
    # strength=0.3,  # controls how much refinement happens
).images[0]

refined_image.save("refined_output_1.jpg")
print("✅ Saved refined image: refined_output_1.jpg")
