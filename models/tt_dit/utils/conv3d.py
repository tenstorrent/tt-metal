# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import collections
import hashlib
from itertools import repeat

import torch
from loguru import logger

import ttnn

from ..layers.module import Module

ALIGNMENT = 32


def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == n, f"{x} must be a tuple of length {n}"
        return tuple(x)
    return tuple(repeat(x, n))


# Mesh-specific blocking table: (h_factor, w_factor, C_in, C_out, kernel) -> blocking
# Blockings swept on BH p150b (130 cores, 13x10 grid).
# WH 4x8 shares (4, 8) entries with BH — needs re-sweep on actual WH hardware.
# See models/tt_dit/tests/models/wan2_2/sweep_results.md for full results.
_MESH_BLOCKINGS = {
    # --- BH Galaxy / WH Galaxy 4x8 (h_factor=4, w_factor=8) — v2: 12x10 grid, correct uncached T ---
    (4, 8, 32, 384, (3, 3, 3)): (32, 128, 1, 16, 16),
    (4, 8, 384, 384, (3, 3, 3)): (128, 128, 1, 4, 4),
    (4, 8, 192, 384, (3, 3, 3)): (96, 128, 1, 8, 4),
    (4, 8, 192, 192, (3, 3, 3)): (96, 96, 1, 4, 16),
    (4, 8, 96, 96, (3, 3, 3)): (96, 96, 1, 8, 8),
    (4, 8, 96, 3, (3, 3, 3)): (96, 32, 1, 8, 16),
    (4, 8, 384, 192, (1, 3, 3)): (96, 96, 1, 16, 16),
    (4, 8, 192, 96, (1, 3, 3)): (192, 96, 1, 4, 32),
    (4, 8, 384, 768, (3, 1, 1)): (192, 256, 1, 4, 4),
    # --- BH Loud Box 2x4 (h_factor=2, w_factor=4) — v2: correct uncached T ---
    (2, 4, 32, 384, (3, 3, 3)): (32, 128, 1, 16, 16),
    (2, 4, 384, 384, (3, 3, 3)): (128, 128, 1, 8, 2),
    (2, 4, 192, 384, (3, 3, 3)): (96, 128, 1, 4, 8),
    (2, 4, 192, 192, (3, 3, 3)): (96, 96, 1, 8, 8),
    (2, 4, 96, 96, (3, 3, 3)): (96, 96, 1, 8, 8),
    (2, 4, 96, 3, (3, 3, 3)): (96, 32, 1, 16, 8),
    (2, 4, 384, 192, (1, 3, 3)): (128, 96, 1, 16, 16),
    (2, 4, 192, 96, (1, 3, 3)): (192, 96, 1, 16, 8),
    (2, 4, 384, 768, (3, 1, 1)): (128, 384, 1, 8, 2),
    # --- BH Galaxy 6U 4x32 (h_factor=4, w_factor=32) — v2: 12x10 grid, verified head-to-head ---
    # Blockings chosen by lowest weighted total across all decoder levels sharing the same key.
    (4, 32, 32, 384, (3, 3, 3)): (32, 128, 1, 16, 2),  # conv_in: 1.61x vs original
    (4, 32, 384, 384, (3, 3, 3)): (96, 128, 1, 16, 2),  # latent+up1: best for up1 (1.41x), ~same latent
    (4, 32, 192, 384, (3, 3, 3)): (96, 128, 1, 16, 2),  # up1 res0: 1.33x vs original
    (4, 32, 192, 192, (3, 3, 3)): (96, 96, 1, 8, 4),  # up2: original is best
    (4, 32, 96, 96, (3, 3, 3)): (96, 96, 1, 8, 8),  # up3: original is best
    (4, 32, 96, 3, (3, 3, 3)): (96, 32, 1, 16, 8),  # conv_out: original is best
    (4, 32, 384, 192, (1, 3, 3)): (192, 96, 1, 32, 4),  # up0+up1 spatial: original is best weighted
    (4, 32, 192, 96, (1, 3, 3)): (192, 96, 1, 4, 8),  # up2 spatial: original is best
    (4, 32, 384, 768, (3, 1, 1)): (192, 384, 1, 16, 2),  # up0+up1 time_conv: 14x vs original default
}

# Fallback table when no mesh-specific entry exists (bh_4x8 defaults).
_DEFAULT_BLOCKINGS = {
    (32, 384, (3, 3, 3)): (32, 64, 1, 4, 16),
    (384, 384, (3, 3, 3)): (128, 64, 1, 4, 8),
    (192, 384, (3, 3, 3)): (96, 64, 1, 4, 8),
    (192, 192, (3, 3, 3)): (96, 96, 1, 8, 8),
    (96, 192, (3, 3, 3)): (96, 64, 1, 4, 4),
    (96, 96, (3, 3, 3)): (96, 96, 1, 16, 4),
    (96, 3, (3, 3, 3)): (96, 32, 1, 16, 8),
    (96, 32, (3, 3, 3)): (96, 32, 1, 16, 8),
    (32, 96, (3, 3, 3)): (32, 96, 1, 16, 8),
    (384, 32, (3, 3, 3)): (32, 32, 1, 4, 1),
    (384, 768, (3, 3, 3)): (96, 96, 1, 8, 4),
    (384, 192, (1, 3, 3)): (128, 64, 1, 32, 4),
    (192, 96, (1, 3, 3)): (192, 96, 1, 8, 4),
    (384, 768, (3, 1, 1)): (128, 384, 1, 8, 8),
    (192, 384, (3, 1, 1)): (192, 128, 1, 1, 32),
}


def get_conv3d_config(in_channels, out_channels, kernel_size, weights_dtype, grid_size, *, h_factor=1, w_factor=1):
    """
    Get optimized Conv3dConfig for a conv3d layer.

    Blockings are mesh-aware: different (h_factor, w_factor) use different spatial tiling
    since per-device tensor shapes depend on how H and W are fractured across devices.
    """
    if weights_dtype == ttnn.float32:
        return ttnn.Conv3dConfig(
            weights_dtype=weights_dtype,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            T_out_block=1,
            W_out_block=1,
            H_out_block=1,
            C_out_block=32,
            C_in_block=32,
            compute_with_storage_grid_size=grid_size,
        )

    mesh_key = (h_factor, w_factor, in_channels, out_channels, kernel_size)
    shape_key = (in_channels, out_channels, kernel_size)
    blocking = _MESH_BLOCKINGS.get(mesh_key) or _DEFAULT_BLOCKINGS.get(shape_key)

    if blocking is None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = in_channels, 32, 1, 1, 1
        logger.warning(
            f"No blocking found for {shape_key} on mesh ({h_factor}x{w_factor}). "
            f"Using default: {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}"
        )
    else:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking

    return ttnn.Conv3dConfig(
        weights_dtype=weights_dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )


def _walk_conv3d_modules(module: Module):
    """Yield every child module that has a conv_config (i.e. conv3d layers)."""
    if hasattr(module, "conv_config"):
        yield module
    for _, child in module.named_children():
        yield from _walk_conv3d_modules(child)


def conv3d_blocking_hash(module: Module) -> str:
    """Build a cache key suffix from C_in_block values of all conv3d layers.

    prepare_conv3d_weights reshapes weights by C_in_block, so cached weights
    are only valid for the same blocking configuration.
    """
    cin_blocks = [str(m.conv_config.C_in_block) for m in _walk_conv3d_modules(module)]
    if not cin_blocks:
        return ""
    return "cin" + hashlib.sha256("_".join(cin_blocks).encode()).hexdigest()[:8]


def count_convs(module: Module) -> int:
    """Count the total number of conv3d modules in a module tree."""
    return sum(1 for _ in _walk_conv3d_modules(module))


def conv_pad_height(tensor_BTHWC, h_factor):
    """
    For Wan2.2, in some parallelism schemes height can't be fractured by the factor.
    This function pads the height to the next multiple of the factor.
    """
    B, T, H, W, C = tensor_BTHWC.shape

    # Calculate padding needed to make H divisible by h_factor
    pad_h = (h_factor - H % h_factor) % h_factor

    if pad_h > 0:
        # Pad height dimension with zeros
        tensor_BTHWC = torch.nn.functional.pad(tensor_BTHWC, (0, 0, 0, 0, 0, pad_h))

    # Return padded tensor and original height for later unpadding
    return tensor_BTHWC, H


def conv_unpad_height(tensor_BTHWC, logical_h):
    """
    For Wan2.2, remove height padding that was added by conv_pad_height.
    """
    B, T, H, W, C = tensor_BTHWC.shape
    # Slice out the original height dimension
    return tensor_BTHWC[:, :, :logical_h, :, :]


def conv_pad_in_channels(tensor):
    C_in = tensor.shape[-1]
    padded_C_in = aligned_channels(C_in)
    if padded_C_in != C_in:
        tensor = torch.nn.functional.pad(tensor, (0, padded_C_in - C_in))
    return tensor


def aligned_channels(channels):
    ALIGN_PAD = ALIGNMENT - channels % ALIGNMENT
    if channels % ALIGNMENT != 0:
        channels = channels + ALIGN_PAD
    return channels
