# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic CPU reference VAE decode of saved TT denoiser latents."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKLFlux2, Flux2Pipeline
from diffusers.image_processor import VaeImageProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    start = time.perf_counter()
    packed = torch.load(args.latents, weights_only=True).float()
    assert list(packed.shape) == [1, 4096, 128] and bool(torch.isfinite(packed).all())
    vae = AutoencoderKLFlux2.from_pretrained(args.checkpoint, subfolder="vae", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        latents = packed.transpose(1, 2).reshape(1, 128, 64, 64)
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        z = Flux2Pipeline._unpatchify_latents(latents * std + mean)
        print("REFERENCE_DECODE_START", list(z.shape), float(z.square().mean().sqrt()), flush=True)
        decoded = vae.decode(z, return_dict=False)[0]
        assert bool(torch.isfinite(decoded).all())
        image = VaeImageProcessor().postprocess(decoded, output_type="pil")[0]
    image.save(args.output / "reference-vae.png")
    result = dict(latents=str(args.latents), latents_sha256=hashlib.sha256(args.latents.read_bytes()).hexdigest(),
                  checkpoint=args.checkpoint, decoder="Diffusers AutoencoderKLFlux2 CPU FP32, no tiling",
                  elapsed_seconds=time.perf_counter() - start, decoded_min=float(decoded.min()),
                  decoded_max=float(decoded.max()), decoded_std=float(decoded.std()),
                  image_sha256=hashlib.sha256((args.output / "reference-vae.png").read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
