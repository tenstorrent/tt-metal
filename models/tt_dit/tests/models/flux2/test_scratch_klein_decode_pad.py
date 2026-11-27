# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: FLUX.2 klein's decoded image at sizes whose latents do not split over the VAE's devices,
against the diffusers VAE in float32 on the CPU decoding the same latents unpadded. Per size of
KLEIN_SIZES (HxW) one eager text-to-image request of 4 steps; prints the PSNR of the device image
against the torch one.
"""

import os

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2

import ttnn

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from .device_params import line_params_flux2_klein


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein], [(4, 8), line_params_flux2_klein]],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_decode_pad(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    sizes = [
        tuple(int(v) for v in s.split("x")) for s in os.environ.get("KLEIN_SIZES", "1040x1024,1056x1024").split(",")
    ]
    checkpoint = model_location_generator("black-forest-labs/FLUX.2-klein-4B")
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=checkpoint,
        resolutions=sorted({(1024, 1024), *((w, h) for h, w in sizes)}),
    )
    torch_vae = AutoencoderKLFlux2.from_pretrained(checkpoint, subfolder="vae", torch_dtype=torch.float32).eval()

    captured = {}
    preprocess = pipeline._vae_decoder.preprocess_and_unpatchify

    def capture(z: torch.Tensor, *, height: int, width: int) -> ttnn.Tensor:
        captured["z"], captured["hw"] = z.clone(), (height, width)
        return preprocess(z, height=height, width=width)

    pipeline._vae_decoder.preprocess_and_unpatchify = capture

    for h, w in sizes:
        (image,) = pipeline(
            prompts=["a lighthouse on a cliff, golden hour"], num_inference_steps=4, seed=0, width=w, height=h
        )
        z, (lh, lw) = captured["z"], captured["hw"]
        mean, std = pipeline._vae_decoder._bn_host
        p = 2
        z = z.float() * std + mean
        z = z.reshape(z.shape[0], lh // p, lw // p, -1, p, p).permute(0, 1, 4, 2, 5, 3).reshape(z.shape[0], lh, lw, -1)
        z = z.to(torch.bfloat16).float().permute(0, 3, 1, 2)
        with torch.no_grad():
            x = torch_vae.decode(z).sample
        ref = ((x[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().numpy().astype(np.float64)
        got = np.asarray(image).astype(np.float64)
        assert got.shape == ref.shape, (got.shape, ref.shape)
        mse = np.mean((got - ref) ** 2)
        edge = np.mean((got[-64:] - ref[-64:]) ** 2)
        print(
            f"DECODE {h}x{w}: PSNR {10 * np.log10(255**2 / mse):.1f} dB, "
            f"last 64 rows {10 * np.log10(255**2 / edge):.1f} dB",
            flush=True,
        )
