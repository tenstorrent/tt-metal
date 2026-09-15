# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side conditioning math for Wan2.2 TI2V-5B image-to-video.

TI2V-5B does not condition on an image by concatenating channels (that is the Wan2.2-14B
I2V scheme) and has no CLIP image encoder — its checkpoint has ``in_channels == out_channels
== 48`` and ``image_dim: null``. Instead it **pins latent frame 0 to the conditioning latent
and gives that frame's tokens a timestep of 0** while every other token sees ``t``.

Everything here is pure torch with no device dependency, so it can be unit-tested against
the reference in seconds without hardware. The reference is
``diffusers/pipelines/wan/pipeline_wan_i2v.py`` ``WanImageToVideoPipeline`` with
``expand_timesteps=True``; the corresponding line numbers in diffusers 0.38.0 are cited on
each function.
"""

from __future__ import annotations

import torch

from ...utils.padding import get_padded_vision_seq_len

__all__ = [
    "blend_condition",
    "denormalize_latents",
    "first_frame_mask",
    "normalize_latents",
    "pad_timesteps_for_sequence_parallel",
    "per_token_timesteps",
]


def first_frame_mask(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``1`` where the latent should be denoised, ``0`` where it is pinned to the condition.

    Shape ``(1, 1, F, h, w)`` with frame 0 zeroed. Mirrors diffusers ``pipeline_wan_i2v.py:461-466``.
    """
    mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)
    mask[:, :, 0] = 0
    return mask


def per_token_timesteps(
    mask: torch.Tensor,
    t: float | torch.Tensor,
    *,
    patch_height: int = 2,
    patch_width: int = 2,
) -> torch.Tensor:
    """Flatten ``mask * t`` to one timestep per transformer token.

    Mirrors diffusers ``pipeline_wan_i2v.py:762``::

        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()

    The strided subsample is the patchify: the transformer sees ``F * (h//pH) * (w//pW)``
    tokens. The resulting order is patch-F major, then H, then W — which is exactly the order
    ``WanTransformer3DModel.preprocess_spatial_input_host`` produces, since it permutes to
    ``(B, patch_F, patch_H, patch_W, ...)`` before flattening.

    Returns a 1-D tensor of length ``N`` (the unpadded token count).
    """
    return (mask[0][0][:, ::patch_height, ::patch_width] * t).flatten()


def pad_timesteps_for_sequence_parallel(
    timesteps_N: torch.Tensor,
    num_devices: int,
    *,
    fill: float | torch.Tensor,
) -> torch.Tensor:
    """Right-pad the per-token timestep vector to the sequence-parallel token count.

    **This deliberately diverges from diffusers, which never pads.** The spatial input is
    padded by ``pad_vision_seq_parallel`` (``utils/padding.py``), which zero-fills. Zero is a
    meaningful timestep — it reads as "fully denoised" — so zero-filling here would change
    the AdaLN modulation of the pad tokens relative to the scalar-timestep path, where every
    token (pad tokens included) sees ``t``. Filling with ``t`` keeps the pad tokens' behaviour
    identical to today's T2V path, which is what makes the all-ones-mask equivalence test
    exact.
    """
    if timesteps_N.ndim != 1:
        msg = f"expected a 1-D per-token timestep vector, got shape {tuple(timesteps_N.shape)}"
        raise ValueError(msg)

    n = timesteps_N.shape[0]
    padded_n = get_padded_vision_seq_len(n, num_devices)
    if padded_n == n:
        return timesteps_N
    pad = torch.full(
        (padded_n - n,),
        float(fill),
        dtype=timesteps_N.dtype,
        device=timesteps_N.device,
    )
    return torch.cat([timesteps_N, pad])


def blend_condition(condition: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(1 - mask) * condition + mask * latents``.

    Mirrors diffusers ``pipeline_wan_i2v.py:758`` (per step) and ``:813-814`` (once more after
    the denoise loop, so frame 0 of the decoded video is exactly the conditioning latent).

    ``condition`` carries a single latent frame (``T == 1``) and broadcasts against the full
    ``T == F`` latent; since ``mask`` is zero only at frame 0, the net effect is that latent
    frame 0 becomes the condition and every other frame is left as the denoised latent.
    """
    return (1 - mask) * condition + mask * latents


def normalize_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Scale raw VAE latents into the transformer's latent space.

    Mathematically ``(x - mean) / std``, but written the way diffusers writes it
    (``pipeline_wan_i2v.py:445-460``) — a multiply by a pre-reciprocated std — because the
    two are **not** bit-identical in floating point (~5e-7 apart at fp32). Matching the
    reference formulation exactly keeps the port bit-for-bit and lets the unit test assert
    equality rather than a tolerance.
    """
    return (latents - mean) * (1.0 / std)


def denormalize_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`normalize_latents`: ``x * std + mean``.

    This is what ``WanVAEDecoderAdapter.decode`` already applies before decoding
    (``models/tt_dit/models/vae/vae_wan2_1.py:2262``), using the same raw (non-reciprocal)
    ``_latents_std``. Kept here so the round-trip can be asserted in a unit test.
    """
    return latents * std + mean
