# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cached CPU reference clips for the audio squeeze tests."""

import json
import os
import sys

import torch

HOP_LENGTH = 800


def ref_dir() -> str:
    return os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "sqz_refs")


def reference_clip(num_latent_frames: int, batch: int):
    """(latents, expected) pair"""
    os.makedirs(ref_dir(), exist_ok=True)
    path = os.path.join(ref_dir(), f"ref_{num_latent_frames}lat_b{batch}.pt")
    if os.path.exists(path):
        blob = torch.load(path)
        return blob["latents"], blob["expected"]
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    weights_dir = os.path.join(os.environ["MINIMAX_H3_MODEL_PATH"], "audio_vae")
    with open(os.path.join(weights_dir, "config.json")) as fh:
        config = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    torch.manual_seed(1)
    waveform = torch.randn(batch, 1, num_latent_frames * HOP_LENGTH) * 0.1
    with torch.no_grad():
        latents = reference.encode(waveform).latent_dist.mode()[..., :num_latent_frames]
        expected = reference.decode(latents).sample
    torch.save({"latents": latents, "expected": expected}, path)
    return latents, expected


if __name__ == "__main__":
    frames, batch = int(sys.argv[1]), int(sys.argv[2])
    lat, exp = reference_clip(frames, batch)
    print(f"latents {tuple(lat.shape)} expected {tuple(exp.shape)}")
