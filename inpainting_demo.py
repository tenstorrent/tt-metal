from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
import os


pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float32
).to("cpu")

# pipe = StableDiffusionXLInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32).to("cpu")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"
generator = torch.Generator(device="cpu").manual_seed(0)

# print("Pipe is: ", pipe)

image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
).images[0]

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save the generated image
output_path = os.path.join(output_dir, "inpainted_image.png")
image.save(output_path)

print(f"✅ Inpainted image saved to: {output_path}")

# Optionally display the image (works in Jupyter notebooks or GUI environments)
try:
    image.show()
    print("🖼️  Image displayed!")
except:
    print("💡 Run in a GUI environment to display the image automatically")
