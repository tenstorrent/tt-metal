# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4.0 text-to-image inference pipeline.

Faithful port of the reference ``pipeline_ideogram4.py`` orchestration, composing
the already-verified tt_dit components:
  * text encoder  — encoders.qwen3vl.Qwen3VlTextEncoder (13-layer tap)
  * denoiser      — models.transformers.transformer_ideogram4.Ideogram4Transformer
                    (a conditional and an unconditional instance — asymmetric CFG)
  * VAE decoder   — models.vae.vae_ideogram4.Ideogram4VAEDecoder
  * sampler       — pipelines.ideogram4.sampler.Ideogram4Sampler

The host-side sequence packing, the per-step CFG blend (v = gw*v_cond +
(1-gw)*v_uncond), the Euler update and the decode (per-channel latent denorm ->
2x2 unpatch -> VAE) mirror the reference exactly. The image latent is sharded on
sequence for SP and the wrapper is per-token, so the denoise loop is SP/TP-ready.
"""

from __future__ import annotations

import torch

import ttnn

from ...models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ...reference.ideogram4.latent_norm import get_latent_norm
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .sampler import Ideogram4Sampler


def unpatchify_latent(z: torch.Tensor, *, grid_h: int, grid_w: int, patch: int) -> torch.Tensor:
    """[B, grid_h*grid_w, patch*patch*ae_ch] -> [B, ae_ch, grid_h*patch, grid_w*patch] (NCHW).

    Matches the reference Ideogram4Pipeline._decode unpatch.
    """
    b = z.shape[0]
    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(b, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    return z.view(b, ae_channels, grid_h * patch, grid_w * patch)


class Ideogram4DecodeStage:
    """The pipeline's decode tail: per-channel latent denorm -> 2x2 unpatch -> VAE decode.

    Split out because it is the pipeline-specific NEW device computation (the denoise
    loop reuses the verified sampler + transformer, the encode reuses the verified
    Qwen3-VL encoder). ``latent_shift``/``latent_scale`` are the per-channel
    (128 = 32 ae-channels x 2x2 patch) constants from the reference latent_norm.
    """

    def __init__(self, vae_decoder: Ideogram4VAEDecoder, *, mesh_device: ttnn.MeshDevice, patch: int = 2) -> None:
        self.vae_decoder = vae_decoder
        self.mesh_device = mesh_device
        self.patch = patch
        shift, scale = get_latent_norm()  # each [128]
        self.latent_shift = bf16_tensor(shift.view(1, 1, -1), device=mesh_device)
        self.latent_scale = bf16_tensor(scale.view(1, 1, -1), device=mesh_device)

    def decode(self, z: ttnn.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
        """z: [B, grid_h*grid_w, 128] denoised latent (device). Returns [B,3,H,W] in [-1,1] (torch)."""
        z = z * self.latent_scale + self.latent_shift  # per-channel denorm (device)

        z_torch = tensor.to_torch(z, mesh_axes=[None, None, None])
        z_nchw = unpatchify_latent(z_torch, grid_h=grid_h, grid_w=grid_w, patch=self.patch)

        tt_z = ttnn.from_torch(
            z_nchw.permute(0, 2, 3, 1),  # NCHW -> NHWC
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
        )
        decoded = self.vae_decoder(tt_z)
        return tensor.to_torch(decoded, mesh_axes=[None, None, None, None]).permute(0, 3, 1, 2)  # NHWC -> NCHW

    @staticmethod
    def to_images(decoded: torch.Tensor) -> torch.Tensor:
        """[-1,1] float -> uint8 [B,H,W,3], matching the reference final step."""
        decoded = decoded.float().clamp(-1.0, 1.0)
        return ((decoded + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)


def cfg_blend(v_cond: ttnn.Tensor, v_uncond: ttnn.Tensor, guidance_weight: float) -> ttnn.Tensor:
    """Asymmetric-CFG velocity blend: v = gw*v_cond + (1-gw)*v_uncond (device)."""
    return v_cond * guidance_weight + v_uncond * (1.0 - guidance_weight)


__all__ = ["Ideogram4DecodeStage", "Ideogram4Sampler", "cfg_blend", "unpatchify_latent"]
