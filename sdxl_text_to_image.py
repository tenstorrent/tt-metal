#!/usr/bin/env python3

"""
Simple Stable Diffusion XL Text-to-Image Generator
Generates an image from a text prompt and saves it as out_ref.png
"""

from diffusers import StableDiffusionXLPipeline
import torch


def generate_image():
    """Generate image from text prompt using SDXL"""

    # Text prompt - modify this to change what gets generated
    prompt = "a majestic dragon flying over a medieval castle at sunset, highly detailed, fantasy art"

    print("🎨 Loading Stable Diffusion XL pipeline...")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    print(f"🎭 Generating image with prompt: '{prompt}'")
    print("⚡ Processing... (this may take a few minutes)")

    # Generate image
    generator = torch.Generator(device=device).manual_seed(42)

    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=8.0,
        generator=generator,
    ).images[0]

    # Save the image
    output_filename = "out_ref.png"
    image.save(output_filename)

    print(f"✅ Image saved as: {output_filename}")

    # Try to display the image
    try:
        image.show()
        print("🖼️  Image displayed!")
    except:
        print("💡 Image saved successfully. Open out_ref.png to view it.")

    return image


if __name__ == "__main__":
    generate_image()
