# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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


# Blocking table: (h_factor, w_factor, C_in, C_out, kernel, H_out, W_out) -> blocking
# Each production (mesh, resolution, layer) combination gets its own entry.
# Blockings are (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).
# See models/tt_dit/tests/models/wan2_2/sweep_results.md for sweep methodology.
_BLOCKINGS = {
    # --- BH Galaxy 6U 4x32, 720p — v3: sliding window + fused tilize+matmul ---
    # conv_in: 32→384 k333, (23,25,7) -> H_out=23, W_out=5
    (4, 32, 32, 384, (3, 3, 3), 23, 5): (32, 128, 1, 16, 2),
    # lat_res + mid_res: 384→384 k333, (23,25,7) -> H_out=23, W_out=5
    (4, 32, 384, 384, (3, 3, 3), 23, 5): (96, 96, 1, 32, 1),
    # up0_tconv: 384→768 k311, (22,23,5) -> H_out=20, W_out=5
    (4, 32, 384, 768, (3, 1, 1), 20, 5): (96, 256, 1, 32, 2),
    # up0_spatial: 384→192 k133, (41,48,12) -> H_out=46, W_out=10
    (4, 32, 384, 192, (1, 3, 3), 46, 10): (96, 96, 1, 16, 4),
    # up1_res0: 192→384 k333, (43,48,12) -> H_out=46, W_out=10
    (4, 32, 192, 384, (3, 3, 3), 46, 10): (96, 128, 1, 16, 2),
    # up1_res: 384→384 k333, (43,48,12) -> H_out=46, W_out=10
    (4, 32, 384, 384, (3, 3, 3), 46, 10): (96, 128, 1, 16, 2),
    # up1_tconv: 384→768 k311, (42,46,10) -> H_out=40, W_out=10
    (4, 32, 384, 768, (3, 1, 1), 40, 10): (192, 384, 1, 16, 2),
    # up1_spatial: 384→192 k133, (81,94,22) -> H_out=92, W_out=20
    (4, 32, 384, 192, (1, 3, 3), 92, 20): (192, 96, 1, 32, 4),
    # up2_res: 192→192 k333, (83,94,22) -> H_out=92, W_out=20
    (4, 32, 192, 192, (3, 3, 3), 92, 20): (96, 96, 1, 32, 2),
    # up2_spatial: 192→96 k133, (81,186,42) -> H_out=184, W_out=40
    (4, 32, 192, 96, (1, 3, 3), 184, 40): (192, 96, 1, 4, 8),
    # up3_res: 96→96 k333, (83,186,42) -> H_out=184, W_out=40
    (4, 32, 96, 96, (3, 3, 3), 184, 40): (96, 96, 1, 16, 4),
    # conv_out: 96→3 k333, (83,186,42) -> H_out=184, W_out=40
    (4, 32, 96, 3, (3, 3, 3), 184, 40): (96, 32, 1, 32, 2),
    # --- BH Galaxy / WH Galaxy 4x8, 720p — v2 ---
    (4, 8, 32, 384, (3, 3, 3), 23, 18): (32, 128, 1, 16, 16),
    (4, 8, 384, 384, (3, 3, 3), 23, 18): (128, 128, 1, 4, 4),
    (4, 8, 192, 384, (3, 3, 3), 46, 38): (96, 128, 1, 8, 4),
    (4, 8, 192, 192, (3, 3, 3), 92, 78): (96, 96, 1, 4, 16),
    (4, 8, 96, 96, (3, 3, 3), 184, 158): (96, 96, 1, 8, 8),
    (4, 8, 96, 3, (3, 3, 3), 184, 158): (96, 32, 1, 8, 16),
    (4, 8, 384, 192, (1, 3, 3), 92, 78): (96, 96, 1, 16, 16),
    (4, 8, 192, 96, (1, 3, 3), 184, 158): (192, 96, 1, 4, 32),
    (4, 8, 384, 768, (3, 1, 1), 40, 40): (192, 256, 1, 4, 4),
    # --- BH Loud Box 2x4, 480p — v2 ---
    (2, 4, 32, 384, (3, 3, 3), 33, 23): (32, 128, 1, 16, 16),
    (2, 4, 384, 384, (3, 3, 3), 33, 23): (128, 128, 1, 8, 2),
    (2, 4, 192, 384, (3, 3, 3), 68, 48): (96, 128, 1, 4, 8),
    (2, 4, 192, 192, (3, 3, 3), 138, 98): (96, 96, 1, 8, 8),
    (2, 4, 96, 96, (3, 3, 3), 278, 198): (96, 96, 1, 8, 8),
    (2, 4, 96, 3, (3, 3, 3), 278, 198): (96, 32, 1, 16, 8),
    (2, 4, 384, 192, (1, 3, 3), 138, 98): (128, 96, 1, 16, 16),
    (2, 4, 192, 96, (1, 3, 3), 278, 198): (192, 96, 1, 16, 8),
    (2, 4, 384, 768, (3, 1, 1), 60, 60): (128, 384, 1, 8, 2),
}

# Fallback table: (C_in, C_out, kernel) -> blocking.
# Used when no exact (mesh, spatial) match exists.
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


def get_conv3d_config(
    in_channels, out_channels, kernel_size, weights_dtype, grid_size, *, h_factor=1, w_factor=1, H_out=0, W_out=0
):
    """
    Get optimized Conv3dConfig for a conv3d layer.

    Lookup chain: exact (mesh, spatial) match → fallback (channel, kernel) match → default.
    Pass h_factor, w_factor, H_out, W_out for best results. When these are not available
    the fallback table is used.
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

    key = (h_factor, w_factor, in_channels, out_channels, kernel_size, H_out, W_out)
    shape_key = (in_channels, out_channels, kernel_size)
    blocking = _BLOCKINGS.get(key) or _DEFAULT_BLOCKINGS.get(shape_key)

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
