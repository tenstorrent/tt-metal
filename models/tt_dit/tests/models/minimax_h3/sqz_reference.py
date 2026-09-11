# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU reference clips for the audio-decoder squeeze experiments: encoder latents of a seeded noise clip and their
reference decode, cached as ``ref_<T>lat_b<B>.pt``. No ttnn import, so it can be precomputed on a host without a
build:

    MINIMAX_H3_MODEL_PATH=... SQZ_REF_DIR=... python -m models.tt_dit.tests.models.minimax_h3.sqz_reference 600 1
"""

import json
import os
import sys
import time

import torch

HOP_LENGTH = 800


def ref_dir() -> str:
    return os.environ.get("SQZ_REF_DIR", os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "sqz_refs"))


def reference_clip(num_latent_frames: int, batch: int):
    """``(latents, expected)``; computed with the pinned diffusers reference on first use, then cached."""
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
    t0 = time.perf_counter()
    with torch.no_grad():
        latents = reference.encode(waveform).latent_dist.mode()[..., :num_latent_frames]
        expected = reference.decode(latents).sample
    print(f"CPU reference for {num_latent_frames} latents x batch {batch}: {time.perf_counter() - t0:.1f} s -> {path}")
    torch.save({"latents": latents, "expected": expected}, path)
    return latents, expected


if __name__ == "__main__":
    frames, batch = int(sys.argv[1]), int(sys.argv[2])
    lat, exp = reference_clip(frames, batch)
    print(f"latents {tuple(lat.shape)} expected {tuple(exp.shape)}")
