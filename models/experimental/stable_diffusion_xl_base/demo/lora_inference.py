from diffusers import DiffusionPipeline
import torch

torch.manual_seed(0)

model_path = "HiFei4869/sd-naruto-model-lora-sdxl"
print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
print("Pipeline Loaded")
pipe.to("cpu")

print("Loading Lora Weights...")
pipe.load_lora_weights(model_path)
print("Lora Weights Loaded")

prompt = "A naruto with green eyes and red legs."
print("Pipeline vae:")
print(pipe.vae)
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("naruto.png")
