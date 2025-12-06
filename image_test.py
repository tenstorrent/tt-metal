#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import argparse
import requests
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from utils.validation_utils import compare_images


def generate_image(
    prompt: str,
    server_url: str = "http://127.0.0.1:8000",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_rescale: float = 0.0,
    seed: int = 14241,
    prompt_2: str = None,
    negative_prompt: str = "cartoon, drawing, low quality, bad quality, distorted, noise, watermark, text, signature",
    negative_prompt_2: str = None,
) -> tuple:
    """
    Send request to server and get generated image

    Args:
        prompt: Text prompt for image generation (primary)
        server_url: Server URL
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        guidance_rescale: Guidance rescale factor (0.0-1.0)
        seed: Random seed for reproducibility
        prompt_2: Secondary prompt for SDXL's second text encoder
        negative_prompt: Negative prompt (primary)
        negative_prompt_2: Negative prompt for SDXL's second text encoder

    Returns:
        Tuple of (PIL Image, inference_time)
    """
    print(f"Sending request to {server_url}...")
    print(f"Prompt: {prompt}")
    if prompt_2:
        print(f"Prompt 2: {prompt_2}")

    request_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "guidance_rescale": guidance_rescale,
        "seed": seed,
        "number_of_images": 1,
    }

    # Add optional dual prompts if provided
    if prompt_2 is not None:
        request_data["prompt_2"] = prompt_2
    if negative_prompt_2 is not None:
        request_data["negative_prompt_2"] = negative_prompt_2

    response = requests.post(
        f"{server_url}/image/generations",
        json=request_data,
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Server error ({response.status_code}): {response.text}")

    data = response.json()
    inference_time = data["inference_time"]
    print(f"Inference completed in {inference_time:.2f}s")

    # Decode image
    base64_img = data["images"][0]
    img_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(img_bytes))

    return image, inference_time


def main():
    parser = argparse.ArgumentParser(description="Test SDXL image generation")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output filename")
    parser.add_argument("--compare", type=str, help="Reference image for comparison")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (12-100)")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale (1.0-20.0)")
    parser.add_argument("--rescale", type=float, default=0.0, help="Guidance rescale factor (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=14241, help="Random seed for reproducibility")
    parser.add_argument("--prompt2", type=str, help="Secondary prompt for SDXL's second text encoder")
    parser.add_argument(
        "--negative",
        type=str,
        default="cartoon, drawing, low quality, bad quality, distorted, noise, watermark, text, signature",
        help="Negative prompt",
    )
    parser.add_argument("--negative2", type=str, help="Negative prompt for SDXL's second text encoder")
    args = parser.parse_args()

    # Check server health first
    try:
        health_response = requests.get(f"{args.server}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Server status: {health_data['status']}")
            print(f"Workers: {health_data['workers_alive']}/{health_data['workers_total']}")
        else:
            print(f"Warning: Server health check failed")
    except Exception as e:
        print(f"Error: Cannot connect to server at {args.server}")
        print(f"Make sure the server is running: ./launch_sdxl_server.sh")
        sys.exit(1)

    # Generate image
    print("")
    try:
        image, inference_time = generate_image(
            args.prompt,
            args.server,
            args.steps,
            args.guidance,
            args.rescale,
            args.seed,
            prompt_2=args.prompt2,
            negative_prompt=args.negative,
            negative_prompt_2=args.negative2,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Save output
    output_path = Path(args.output)
    image.save(output_path)
    print(f"Image saved to: {output_path}")

    # Compare with reference if provided
    if args.compare:
        ref_path = Path(args.compare)
        if not ref_path.exists():
            print(f"Warning: Reference image not found: {ref_path}")
        else:
            print(f"\nComparing with reference: {ref_path}")
            ref_image = Image.open(ref_path)

            metrics = compare_images(image, ref_image)
            print(f"  MSE: {metrics['mse']:.2f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  Similar: {metrics['similar']}")

            if metrics["similar"]:
                print("✓ Images are similar (SSIM >= 0.9)")
                sys.exit(0)
            else:
                print("✗ Images differ significantly (SSIM < 0.9)")
                print("Note: Some variation is expected with generative models.")
                print("      SSIM > 0.85 indicates similar quality.")
                sys.exit(1 if metrics["ssim"] < 0.85 else 0)


if __name__ == "__main__":
    main()
