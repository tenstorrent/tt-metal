# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein latent geometry, position ids, noise and host-side token packing.

Host-side and pure torch: every value here is part of the checkpoint's numerical contract and is compared
bit-exactly against the diffusers ``Flux2KleinPipeline`` (``pipeline_flux2_klein.py``, cited as ``P``) with the
BFL ``flux2`` sources (``sampling.py`` ``SA``, ``autoencoder.py`` ``BAE``) as the second citation, so nothing in
this module may be reassociated, re-typed or moved to device. Neither reference is imported here.

Sequence contract (``P:846-848``, ``SA:292-293``): the DiT's image stream is ``[latents | reference_0 |
reference_1 | ...]`` in row-major ``(h, w)`` token order per image; the prompt is a separate stream of exactly
``FLUX2_KLEIN_TEXT_LENGTH`` rows. Ids are 4-axis ``(t, h, w, l)`` int64 built with ``torch.cartesian_prod``:
latents ``(0, h, w, 0)`` (``P:287-316``), reference ``i`` ``(10 + 10 i, h, w, 0)`` (``P:320-367``, ``scale=10``),
text ``(0, 0, 0, l)`` (``P:267-283``). Tokens are 128 = 32 VAE channels x 2x2 patch channels, packed as
``c * 4 + py * 2 + px`` (``P:371-376``, ``BAE:318-323``).

dtypes: the image state is fp32 holding bf16-valued numbers (the reference draws the noise in bf16, ``P:503``
``randn_tensor(..., dtype=prompt_embeds.dtype)``), reference tokens are bf16, every id is int64.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

# transformer/config.json in_channels: 32 VAE channels x 4 patch positions.
FLUX2_KLEIN_IN_CHANNELS = 128
FLUX2_KLEIN_LATENT_CHANNELS = 128
# vae/config.json: latent_channels 32, patch_size [2, 2], 8x spatial compression -> 16 px per token.
FLUX2_KLEIN_VAE_CHANNELS = 32
FLUX2_KLEIN_PATCH = 2
FLUX2_KLEIN_DOWNSAMPLING_RATIO = 16
# P:204 tokenizer_max_length: the prompt stream is always 512 rows (padded positions are real tokens to the DiT).
FLUX2_KLEIN_TEXT_LENGTH = 512
# A port limit (the reference has no cap on condition images).
FLUX2_KLEIN_MAX_REFERENCES = 10
# P:322 _prepare_image_ids(scale=10): reference i sits at t = scale + scale * i.
FLUX2_KLEIN_REFERENCE_T_SCALE = 10
# Tile height. Kept as a plain int so this module stays free of ``ttnn``; it equals ``ttnn.TILE_SIZE``.
FLUX2_KLEIN_TILE = 32
# Distinct ``(seed, h, w)`` draws :func:`make_noise_cached` keeps (a repeated request pays the copy, not the draw).
FLUX2_KLEIN_NOISE_CACHE_SIZE = 8


# --- latent geometry ---------------------------------------------------------------------------------------------


def latent_shape(height: int, width: int) -> tuple[int, int]:
    """``(h, w)`` token grid of a ``height x width`` image (``P:493-496``: ``height // 16`` after the pipeline has
    floored both sides to the 16-px grid; a side off the grid is an error here, never silently floored)."""
    if height % FLUX2_KLEIN_DOWNSAMPLING_RATIO or width % FLUX2_KLEIN_DOWNSAMPLING_RATIO:
        raise ValueError(f"width and height must be multiples of {FLUX2_KLEIN_DOWNSAMPLING_RATIO}")
    return height // FLUX2_KLEIN_DOWNSAMPLING_RATIO, width // FLUX2_KLEIN_DOWNSAMPLING_RATIO


def latent_ids(h: int, w: int) -> Tensor:
    """``(h*w, 4)`` int64 ids ``(0, row, col, 0)`` in row-major token order (``P:306-313`` ``_prepare_latent_ids``)."""
    return torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1))


def reference_ids(h: int, w: int, index: int) -> Tensor:
    """``(h*w, 4)`` int64 ids of reference ``index`` (0-based): ``(10 + 10 * index, row, col, 0)``
    (``P:353-360`` ``_prepare_image_ids``: ``t_coords = [scale + scale * t for t in arange(N)]``)."""
    t = FLUX2_KLEIN_REFERENCE_T_SCALE + FLUX2_KLEIN_REFERENCE_T_SCALE * torch.arange(index, index + 1)
    return torch.cartesian_prod(t, torch.arange(h), torch.arange(w), torch.arange(1))


def text_ids(length: int) -> Tensor:
    """``(length, 4)`` int64 ids ``(0, 0, 0, position)`` (``P:273-279`` ``_prepare_text_ids``)."""
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length))


def latent_grid_from_ids(ids: Tensor) -> tuple[int, int]:
    """``(h, w)`` recovered from latent ids the way ``_unpack_latents_with_ids`` does without a given size
    (``P:413-414``: ``max(h_ids) + 1``, ``max(w_ids) + 1``)."""
    return int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1


# --- noise -------------------------------------------------------------------------------------------------------


def make_noise(seed: int, h: int, w: int) -> Tensor:
    """The initial latents of ``prepare_latents`` (``P:496-510``) for a CPU generator: ``randn_tensor`` is
    ``torch.randn((1, 128, h, w), generator, dtype=bf16)`` (``utils/torch_utils.py:197``), then ``_pack_latents``
    (``P:389-397``: ``reshape(1, 128, h*w).permute(0, 2, 1)``). Returned as ``[h*w, 128]`` fp32 holding the bf16
    values (the device state slot is fp32; the typecast to bf16 is exact)."""
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn((1, FLUX2_KLEIN_IN_CHANNELS, h, w), generator=generator, dtype=torch.bfloat16)
    packed = latents.reshape(1, FLUX2_KLEIN_IN_CHANNELS, h * w).permute(0, 2, 1)
    return packed[0].float().contiguous()


@functools.lru_cache(maxsize=FLUX2_KLEIN_NOISE_CACHE_SIZE)
def _make_noise_cached(seed: int, h: int, w: int) -> Tensor:
    return make_noise(seed, h, w)


def make_noise_cached(seed: int, h: int, w: int) -> Tensor:
    """:func:`make_noise` memoised over the last :data:`FLUX2_KLEIN_NOISE_CACHE_SIZE` distinct ``(seed, h, w)``
    (``functools.lru_cache``: thread-safe, process-wide). The bf16 ``randn`` of ``128 x h x w`` costs 20-60 ms on one
    host thread (1K / 2K); a hit costs the copy of the cached tensor (``clone``, about 1 ms at 2K) that every call
    returns, so the caller owns its tensor. Same values as :func:`make_noise` bit for bit (the generator is seeded
    per call; nothing depends on the thread or the call order)."""
    return _make_noise_cached(int(seed), int(h), int(w)).clone()


def noise_cache_info() -> functools._CacheInfo:
    """``hits / misses / maxsize / currsize`` of :func:`make_noise_cached`."""
    return _make_noise_cached.cache_info()


def noise_cache_clear() -> None:
    """Drop every memoised draw (tests)."""
    _make_noise_cached.cache_clear()


# --- 2x2 patchify / unpatchify (P:365-385, BAE:318-334) ----------------------------------------------------------


def patchify_latents(latents: Tensor) -> Tensor:
    """``(32, 2h, 2w)`` VAE latents -> ``(h*w, 128)`` tokens: ``_patchify_latents`` (``P:367-371``, token channel
    ``c*4 + py*2 + px``) followed by ``_pack_latents`` (``P:389-397``). The BatchNorm sits between the two in the
    reference (``P:466-477``); it is per token channel, so it commutes with this function."""
    channels, height, width = latents.shape
    p = FLUX2_KLEIN_PATCH
    if channels != FLUX2_KLEIN_VAE_CHANNELS or height % p or width % p:
        raise ValueError(f"latents of shape {tuple(latents.shape)} are not (32, 2h, 2w)")
    z = latents.view(1, channels, height // p, p, width // p, p).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(1, channels * p * p, height // p, width // p)
    return z.reshape(1, channels * p * p, (height // p) * (width // p)).permute(0, 2, 1)[0]


def unpatchify_tokens(tokens: Tensor, h: int, w: int) -> Tensor:
    """Inverse of :func:`patchify_latents`: ``(h*w, 128)`` tokens in ``latent_ids`` order -> ``(32, 2h, 2w)``
    (``_unpack_latents_with_ids`` with row-major ids, ``P:419-424``, then ``_unpatchify_latents``, ``P:376-380``)."""
    if tokens.shape != (h * w, FLUX2_KLEIN_LATENT_CHANNELS):
        raise ValueError(f"tokens of shape {tuple(tokens.shape)} do not fill a {h}x{w} grid of 128 channels")
    p = FLUX2_KLEIN_PATCH
    z = tokens.view(h, w, FLUX2_KLEIN_LATENT_CHANNELS).permute(2, 0, 1).unsqueeze(0)
    z = z.reshape(1, FLUX2_KLEIN_VAE_CHANNELS, p, p, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(FLUX2_KLEIN_VAE_CHANNELS, h * p, w * p)


# --- BatchNorm (P:470-477, P:911-916; BAE:304-312) ---------------------------------------------------------------


def bn_normalize(tokens: Tensor, running_mean: Tensor, running_var: Tensor, eps: float = 1e-4) -> Tensor:
    """``_encode_vae_image`` (``P:470-477``): ``(z - mean) / sqrt(var + eps)`` per token channel, the statistics
    cast to the tokens' dtype after the sqrt, exactly as the reference orders it."""
    mean = running_mean.reshape(1, -1).to(tokens.dtype)
    std = torch.sqrt(running_var.reshape(1, -1) + eps).to(tokens.dtype)
    return (tokens - mean) / std


def bn_denormalize(tokens: Tensor, running_mean: Tensor, running_var: Tensor, eps: float = 1e-4) -> Tensor:
    """The decode side (``P:911-916``): ``z * sqrt(var + eps) + mean`` with the same casts."""
    mean = running_mean.reshape(1, -1).to(tokens.dtype)
    std = torch.sqrt(running_var.reshape(1, -1) + eps).to(tokens.dtype)
    return tokens * std + mean


# --- tile padding (device layout only; nothing here changes what the model computes) ------------------------------


def padded_length(length: int, sp_factor: int = 1, tile: int = FLUX2_KLEIN_TILE) -> int:
    """``length`` rounded up to a multiple of ``sp_factor * tile`` so a stream fractures evenly over SP into
    tile-aligned rows per device."""
    if sp_factor < 1 or tile < 1:
        raise ValueError(f"sp_factor and tile must be positive, got {sp_factor}, {tile}")
    alignment = sp_factor * tile
    return ((length + alignment - 1) // alignment) * alignment


def pad_rows(rows: Tensor, length: int) -> tuple[Tensor, Tensor]:
    """Zero-pad ``rows`` along dim 0 to ``length``. Returns ``(padded, mask)`` with ``mask[i]`` True for every
    input row and False for the appended ones."""
    if length < rows.shape[0]:
        raise ValueError(f"cannot pad {rows.shape[0]} rows down to {length}")
    pad = length - rows.shape[0]
    mask = torch.cat([torch.ones(rows.shape[0], dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
    if pad == 0:
        return rows, mask
    padding = rows.new_zeros((pad, *rows.shape[1:]))
    return torch.cat([rows, padding]), mask


def pad_segment(rows: Tensor, sp_factor: int = 1, tile: int = FLUX2_KLEIN_TILE) -> tuple[Tensor, Tensor]:
    """:func:`pad_rows` to :func:`padded_length` of the segment."""
    return pad_rows(rows, padded_length(rows.shape[0], sp_factor, tile))


__all__ = [
    "FLUX2_KLEIN_DOWNSAMPLING_RATIO",
    "FLUX2_KLEIN_IN_CHANNELS",
    "FLUX2_KLEIN_LATENT_CHANNELS",
    "FLUX2_KLEIN_MAX_REFERENCES",
    "FLUX2_KLEIN_NOISE_CACHE_SIZE",
    "FLUX2_KLEIN_PATCH",
    "FLUX2_KLEIN_REFERENCE_T_SCALE",
    "FLUX2_KLEIN_TEXT_LENGTH",
    "FLUX2_KLEIN_TILE",
    "FLUX2_KLEIN_VAE_CHANNELS",
    "bn_denormalize",
    "bn_normalize",
    "latent_grid_from_ids",
    "latent_ids",
    "latent_shape",
    "make_noise",
    "make_noise_cached",
    "noise_cache_clear",
    "noise_cache_info",
    "pad_rows",
    "pad_segment",
    "padded_length",
    "patchify_latents",
    "reference_ids",
    "text_ids",
    "unpatchify_tokens",
]
