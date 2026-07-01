# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# VAE encoder posterior on device (mirrors ref/cond_vae_encode.DiagonalGaussianDistribution).

from __future__ import annotations

import ttnn

from models.experimental.hunyuan_image_3_0.ref.vae.encoder import Z_CHANNELS


def diagonal_gaussian_sample_tt(
    parameters_bthwc: ttnn.Tensor,
    *,
    mesh_device,
    deterministic: bool = False,
    seed: int | None = None,
) -> ttnn.Tensor:
    """Sample latent ``[B,T,H,W,Z]`` from encoder head ``[B,T,H,W,2*Z]`` on device."""
    shape = parameters_bthwc.shape
    bsz, t_len, h, w, c = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4])
    half = c // 2
    mean = ttnn.slice(parameters_bthwc, [0, 0, 0, 0, 0], [bsz, t_len, h, w, half])
    logvar = ttnn.slice(parameters_bthwc, [0, 0, 0, 0, half], [bsz, t_len, h, w, c])
    ttnn.deallocate(parameters_bthwc, force=False)
    logvar = ttnn.clamp(logvar, -30.0, 20.0)
    if deterministic:
        ttnn.deallocate(logvar, force=False)
        return mean
    std = ttnn.exp(ttnn.multiply(logvar, 0.5))
    ttnn.deallocate(logvar, force=False)
    noise = ttnn.randn(
        (bsz, t_len, h, w, half),
        device=mesh_device,
        dtype=mean.dtype,
        layout=mean.layout,
        seed=seed if seed is not None else 0,
    )
    scaled = ttnn.multiply(std, noise)
    ttnn.deallocate(std)
    ttnn.deallocate(noise)
    return ttnn.add(mean, scaled)


def apply_vae_latent_scaling_tt(
    latents_bthwc: ttnn.Tensor,
    *,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
) -> ttnn.Tensor:
    """``(z - shift) * scale`` on device."""
    out = latents_bthwc
    if shift_factor:
        out = ttnn.subtract(out, float(shift_factor))
    if scaling_factor:
        out = ttnn.multiply(out, float(scaling_factor))
    if out is not latents_bthwc:
        ttnn.deallocate(latents_bthwc, force=False)
    return out


def squeeze_temporal_tt(latents_bthwc: ttnn.Tensor) -> ttnn.Tensor:
    """``[B,T,H,W,C]`` -> ``[B,1,H,W,C]`` when ``T==1``, else slice ``T=0``."""
    bsz, t_len, h, w, c = (int(latents_bthwc.shape[i]) for i in range(5))
    if t_len == 1:
        return latents_bthwc
    out = ttnn.slice(latents_bthwc, [0, 0, 0, 0, 0], [bsz, 1, h, w, c])
    ttnn.deallocate(latents_bthwc, force=False)
    return out


def latent_bthwc_to_patch_input(latents_bthwc: ttnn.Tensor) -> tuple[ttnn.Tensor, int, int, int]:
    """``[B,1,H,W,Z]`` BTHWC -> flat NHWC ``[1,1,B*H*W,Z]`` for ``HunyuanTtUNetDown``."""
    bsz, t_len, h, w, c = (int(latents_bthwc.shape[i]) for i in range(5))
    assert c == Z_CHANNELS, f"expected Z={Z_CHANNELS}, got {c}"
    if t_len != 1:
        latents_bthwc = squeeze_temporal_tt(latents_bthwc)
        bsz, t_len, h, w, c = (int(latents_bthwc.shape[i]) for i in range(5))
    flat = ttnn.reshape(latents_bthwc, [1, 1, bsz * h * w, c])
    return flat, bsz, h, w
