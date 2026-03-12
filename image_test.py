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


# ---------------------------------------------------------------------------
# Per-model defaults
# ---------------------------------------------------------------------------

_SDXL_DEFAULTS = {
    "steps": 50,
    "guidance": 5.0,
    "seed": 14241,
    "negative": "cartoon, drawing, low quality, bad quality, distorted, noise, watermark, text, signature",
    "output": "output_sdxl.jpg",
}

_SD35_DEFAULTS = {
    "steps": 28,
    "guidance": 3.5,
    "seed": 42,
    "negative": "",
    "output": "output_sd35.jpg",
}


def generate_image(
    prompt: str,
    model: str = "sdxl",
    server_url: str = "http://127.0.0.1:8000",
    num_inference_steps: int = None,
    guidance_scale: float = None,
    seed: int = None,
    negative_prompt: str = None,
    # SDXL-only fields
    guidance_rescale: float = 0.0,
    prompt_2: str = None,
    negative_prompt_2: str = None,
) -> tuple:
    """Send a generation request to the server and return the decoded image.

    Args:
        prompt: Primary text prompt.
        model: 'sdxl' or 'sd35' — controls which fields are included in the
               request body and which defaults are applied.
        server_url: Base URL of the running inference server.
        num_inference_steps: Denoising steps (model-specific default if None).
        guidance_scale: CFG scale (model-specific default if None).
        seed: Random seed for reproducibility (model-specific default if None).
        negative_prompt: Negative prompt (model-specific default if None).
        guidance_rescale: SDXL-only: guidance rescale factor (0.0–1.0).
        prompt_2: SDXL-only: secondary prompt for second text encoder.
        negative_prompt_2: SDXL-only: secondary negative prompt.

    Returns:
        3-tuple: (PIL Image, inference_time_seconds, model_label_string)
    """
    defaults = _SDXL_DEFAULTS if model == "sdxl" else _SD35_DEFAULTS

    # Apply per-model defaults for parameters not explicitly provided
    if num_inference_steps is None:
        num_inference_steps = defaults["steps"]
    if guidance_scale is None:
        guidance_scale = defaults["guidance"]
    if seed is None:
        seed = defaults["seed"]
    if negative_prompt is None:
        negative_prompt = defaults["negative"]

    print(f"Sending request to {server_url}...")
    print(f"Prompt: {prompt}")
    if model == "sdxl" and prompt_2:
        print(f"Prompt 2: {prompt_2}")

    request_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }

    # Include SDXL-specific fields only in SDXL mode
    if model == "sdxl":
        request_data["guidance_rescale"] = guidance_rescale
        if prompt_2 is not None:
            request_data["prompt_2"] = prompt_2
        if negative_prompt_2 is not None:
            request_data["negative_prompt_2"] = negative_prompt_2

    response = requests.post(
        f"{server_url}/image/generations",
        json=request_data,
        timeout=600,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Server error ({response.status_code}): {response.text}")

    data = response.json()
    inference_time = data["inference_time"]
    server_model = data.get("model", "unknown")

    print(f"Inference completed in {inference_time:.2f}s (server model: {server_model})")

    # Decode first image from base64
    base64_img = data["images"][0]
    img_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(img_bytes))

    return image, inference_time, server_model


def main():
    parser = argparse.ArgumentParser(description="Test image generation against the unified inference server")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl",
        choices=["sdxl", "sd35"],
        help="Model to target: 'sdxl' (default) or 'sd35'",
    )
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: output_<model>.jpg)")
    parser.add_argument("--compare", type=str, help="Reference image path for quality comparison")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--negative",
        type=str,
        default=None,
        help="Negative prompt (uses model-specific default if not set)",
    )
    # SDXL-specific arguments (ignored when --model sd35)
    parser.add_argument("--rescale", type=float, default=0.0, help="[SDXL] Guidance rescale factor (0.0-1.0)")
    parser.add_argument("--prompt2", type=str, help="[SDXL] Secondary prompt for second text encoder")
    parser.add_argument("--negative2", type=str, help="[SDXL] Negative prompt for second text encoder")

    args = parser.parse_args()

    # Determine output filename
    defaults = _SDXL_DEFAULTS if args.model == "sdxl" else _SD35_DEFAULTS
    output_file = args.output if args.output is not None else defaults["output"]

    # ---------------------------------------------------------------------------
    # Health check
    # ---------------------------------------------------------------------------
    try:
        health_response = requests.get(f"{args.server}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            server_model = health_data.get("model", "unknown")
            print(f"Server status: {health_data['status']}")
            print(f"Server model:  {server_model}")
            print(f"Workers: {health_data['workers_alive']}/{health_data['workers_total']}")

            # Warn if the server model does not match the requested model
            model_map = {"sdxl": "SDXL", "sd35": "SD3.5 Large"}
            expected_label = model_map.get(args.model, args.model)
            if server_model != expected_label:
                print(
                    f"WARNING: --model {args.model} expects server model '{expected_label}' "
                    f"but server reports '{server_model}'. "
                    f"Results may be unexpected."
                )
        else:
            print("Warning: Server health check returned non-200 status")
    except Exception as e:
        print(f"Error: Cannot connect to server at {args.server}: {e}")
        print("Make sure the server is running: ./launch_server.sh --model <sdxl|sd35>")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Generate image
    # ---------------------------------------------------------------------------
    print("")
    try:
        image, inference_time, server_model_label = generate_image(
            prompt=args.prompt,
            model=args.model,
            server_url=args.server,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            negative_prompt=args.negative,
            guidance_rescale=args.rescale,
            prompt_2=args.prompt2,
            negative_prompt_2=args.negative2,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------------------
    output_path = Path(output_file)
    image.save(output_path)
    print(f"Image saved to: {output_path}")

    # ---------------------------------------------------------------------------
    # Optional reference comparison
    # ---------------------------------------------------------------------------
    if args.compare:
        ref_path = Path(args.compare)
        if not ref_path.exists():
            print(f"Warning: Reference image not found: {ref_path}")
        else:
            print(f"\nComparing with reference: {ref_path}")
            ref_image = Image.open(ref_path)
            metrics = compare_images(image, ref_image)
            print(f"  MSE:     {metrics['mse']:.2f}")
            print(f"  SSIM:    {metrics['ssim']:.4f}")
            print(f"  Similar: {metrics['similar']}")

            if metrics["similar"]:
                print("Images are similar (SSIM >= 0.9)")
                sys.exit(0)
            else:
                print("Images differ significantly (SSIM < 0.9)")
                print("Note: Some variation is expected with generative models.")
                print("      SSIM > 0.85 indicates similar quality.")
                sys.exit(1 if metrics["ssim"] < 0.85 else 0)


if __name__ == "__main__":
    main()
