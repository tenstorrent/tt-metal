# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Patchify / unpatchify.

Wan's patch embedding is a Conv3d whose stride equals its kernel, so it is a linear map
over non-overlapping patches: host rearrange plus a Linear. `to_ndhwc` serves the optional
ttnn conv3d path instead, which runs the conv on device and skips the host rearrange.

The two orders differ -- the conv weight contracts over (channels, p_t, p_h, p_w) while
proj_out emits (p_t, p_h, p_w, channels).
"""

from __future__ import annotations

import numpy as np


def grid_size(latent_shape: tuple, patch_size: tuple) -> tuple[int, int, int]:
    _, _, frames, height, width = latent_shape
    p_t, p_h, p_w = patch_size
    return frames // p_t, height // p_h, width // p_w


def patch_features(in_channels: int, patch_size: tuple) -> int:
    p_t, p_h, p_w = patch_size
    return in_channels * p_t * p_h * p_w


def patchify(latent: np.ndarray, patch_size: tuple) -> np.ndarray:
    """(B, C, F, H, W) -> (B, 1, S, C*p_t*p_h*p_w), in the conv weight's contraction order."""
    batch, channels, frames, height, width = latent.shape
    p_t, p_h, p_w = patch_size
    if frames % p_t or height % p_h or width % p_w:
        raise ValueError(f"latent {latent.shape[2:]} is not divisible by patch {patch_size}")

    ppf, pph, ppw = frames // p_t, height // p_h, width // p_w
    x = latent.reshape(batch, channels, ppf, p_t, pph, p_h, ppw, p_w)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)  # token order f,h,w; features channel-major
    return np.ascontiguousarray(x.reshape(batch, 1, ppf * pph * ppw, channels * p_t * p_h * p_w))


def patchify_output_order(latent: np.ndarray, patch_size: tuple) -> np.ndarray:
    """(B, C, F, H, W) -> (B, 1, S, p_t*p_h*p_w*C): the order proj_out emits.

    Use on the flow-matching target so the loss needs no permutation on device.
    """
    batch, channels, frames, height, width = latent.shape
    p_t, p_h, p_w = patch_size
    ppf, pph, ppw = frames // p_t, height // p_h, width // p_w
    x = latent.reshape(batch, channels, ppf, p_t, pph, p_h, ppw, p_w)
    x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)
    return np.ascontiguousarray(x.reshape(batch, 1, ppf * pph * ppw, p_t * p_h * p_w * channels))


def unpatchify(tokens: np.ndarray, patch_size: tuple, grid: tuple, out_channels: int) -> np.ndarray:
    """(B, 1, S, p_t*p_h*p_w*C) -> (B, C, F, H, W), matching proj_out's channel-minor order."""
    batch = tokens.shape[0]
    ppf, pph, ppw = grid
    p_t, p_h, p_w = patch_size
    x = tokens.reshape(batch, ppf, pph, ppw, p_t, p_h, p_w, out_channels)
    x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
    return np.ascontiguousarray(x.reshape(batch, out_channels, ppf * p_t, pph * p_h, ppw * p_w))


def conv3d_weight_to_linear(weight: np.ndarray) -> np.ndarray:
    """Checkpoint (dim, C, p_t, p_h, p_w) -> ttml linear weight (1, 1, dim, C*p_t*p_h*p_w)."""
    out_dim = weight.shape[0]
    return np.ascontiguousarray(weight.reshape(out_dim, -1)[None, None])


def to_ndhwc(latent: np.ndarray) -> np.ndarray:
    """(B, C, F, H, W) -> (B, F, H, W, C), the activation layout ttnn conv3d expects.

    No companion weight helper: ttnn reorders the checkpoint weight itself, into a blocked
    permutation only prepare_conv3d_weights produces consistently.
    """
    return np.ascontiguousarray(latent.transpose(0, 2, 3, 4, 1))
