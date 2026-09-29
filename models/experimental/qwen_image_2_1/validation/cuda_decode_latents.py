# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Decode saved TT and CUDA Qwen Image 2.1 latents with the pinned CUDA VAE.

This is a validation boundary, not a TT VAE implementation. The CUDA latent
must reproduce the reference pipeline image before comparing the TT latent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from models.experimental.qwen_image_2_1.checkpoint import MODEL_ID, MODEL_REVISION


def _decode(vae, processor, packed: torch.Tensor, height: int, width: int):
    latents = QwenImage21Pipeline._unpack_latents(packed, height, width, 16)
    latents = latents.to(device="cuda", dtype=vae.dtype)
    mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=vae.dtype).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda", dtype=vae.dtype).view(1, 64, 1, 1, 1)
    image = vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    return processor.postprocess(image, output_type="pil")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-reference", type=Path, required=True)
    parser.add_argument("--tt-latents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, help="zero-based denoising step; defaults to the final step")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA VAE validation requires a CUDA GPU")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.cuda_reference / "manifest.json").read_text())
    height, width = manifest["height"], manifest["width"]
    step = manifest["steps"] - 1 if args.step is None else args.step
    if step < 0 or step >= manifest["steps"]:
        parser.error("--step is outside the reference denoising schedule")
    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        checkpoint, subfolder="vae", dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    vae.eval()
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    cuda_latents = torch.load(
        args.cuda_reference / f"step_{step:03d}/scheduler/updated_latents.pt",
        map_location="cpu",
        weights_only=True,
    )
    tt_latents = torch.load(args.tt_latents, map_location="cpu", weights_only=True)
    if cuda_latents.shape != tt_latents.shape:
        raise ValueError(f"latent shapes differ: {cuda_latents.shape} vs {tt_latents.shape}")
    with torch.inference_mode():
        cuda_image = _decode(vae, processor, cuda_latents, height, width)
        tt_image = _decode(vae, processor, tt_latents, height, width)
    cuda_image.save(args.output_dir / "cuda_redecoded.png")
    tt_image.save(args.output_dir / "tt_latents_cuda_vae.png")
    cuda_pixels = np.asarray(cuda_image)
    tt_pixels = np.asarray(tt_image)
    pixel_difference = np.abs(tt_pixels.astype(np.int16) - cuda_pixels.astype(np.int16))
    latent_difference = tt_latents.float() - cuda_latents.float()
    report = {
        "model_revision": MODEL_REVISION,
        "step": step,
        "height": height,
        "width": width,
        "tt_vs_cuda_pixel_mean_abs_error": float(pixel_difference.mean()),
        "tt_vs_cuda_pixel_max_abs_error": int(pixel_difference.max()),
        "tt_vs_cuda_latent_relative_rms_error": float(
            latent_difference.square().mean().sqrt() / cuda_latents.float().square().mean().sqrt()
        ),
        "tt_vs_cuda_latent_max_abs_error": float(latent_difference.abs().max()),
    }
    if step == manifest["steps"] - 1:
        original = np.asarray(Image.open(args.cuda_reference / "reference.png"))
        reference_difference = np.abs(cuda_pixels.astype(np.int16) - original.astype(np.int16))
        report["cuda_redecode_exact_fraction"] = float(np.mean(reference_difference == 0))
        report["cuda_redecode_max_abs_error"] = int(reference_difference.max())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
