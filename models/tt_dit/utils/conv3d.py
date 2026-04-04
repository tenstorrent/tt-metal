# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import collections
import hashlib
import math
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


def compute_decoder_stage_dims(target_height, target_width, h_factor, w_factor, num_stages=3):
    """Compute per-device (H_out, W_out) at each VAE decoder stage.

    Returns a list of (H, W) tuples from latent resolution to full resolution.
    Height is rounded up (padded), width is rounded down (fractured evenly).
    """
    vae_scale = 2**num_stages  # 8 for 3 upsample stages
    # Height rounds up to handle fractured padding; width rounds down (exact fracturing).
    h_dev = math.ceil(target_height / vae_scale / h_factor) * vae_scale
    w_dev = (target_width // vae_scale // w_factor) * vae_scale
    stages = [(h_dev // vae_scale, w_dev // vae_scale)]
    for _ in range(num_stages):
        prev_h, prev_w = stages[-1]
        stages.append((prev_h * 2, prev_w * 2))
    return stages


# Blocking table: (h_factor, w_factor, C_in, C_out, kernel, H_out, W_out) -> blocking
# Each production (mesh, resolution, layer) combination gets its own entry.
# Blockings are (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).
# See models/tt_dit/tests/models/wan2_2/sweep_results.md for sweep methodology.
_BLOCKINGS = {
    # --- BH Galaxy 6U 4x32, 720p — v4: T_out_block > 1 + joint spatial sweep ---
    # conv_in: 32→384 k333, T=1 optimal (small shape)
    (4, 32, 32, 384, (3, 3, 3), 23, 5): (32, 128, 1, 16, 2),
    # lat_res + mid_res: 384→384 k333, T=1 optimal (small shape)
    (4, 32, 384, 384, (3, 3, 3), 23, 5): (96, 96, 1, 32, 1),
    # up0_tconv: 384→768 k311, T=3 gives -8%
    (4, 32, 384, 768, (3, 1, 1), 23, 5): (128, 256, 3, 8, 4),
    # up0_spatial: 384→192 k133, T=1 (kT=1, no T blocking possible)
    (4, 32, 384, 192, (1, 3, 3), 46, 10): (96, 96, 1, 16, 4),
    # up1_res0: 192→384 k333, T=3 gives -4%
    (4, 32, 192, 384, (3, 3, 3), 46, 10): (96, 128, 3, 8, 4),
    # up1_res: 384→384 k333, T=1 optimal (multi C_in block, L1 pressure at higher T)
    (4, 32, 384, 384, (3, 3, 3), 46, 10): (96, 128, 1, 16, 2),
    # up1_tconv: 384→768 k311, T=3 gives -6%
    (4, 32, 384, 768, (3, 1, 1), 46, 10): (192, 384, 3, 16, 2),
    # up1_spatial: 384→192 k133, T=1 (kT=1)
    (4, 32, 384, 192, (1, 3, 3), 92, 20): (192, 96, 1, 32, 4),
    # up2_res: 192→192 k333, T=3 gives -13%
    (4, 32, 192, 192, (3, 3, 3), 92, 20): (96, 96, 3, 8, 4),
    # up2_spatial: 192→96 k133, T=1 (kT=1)
    (4, 32, 192, 96, (1, 3, 3), 184, 40): (192, 96, 1, 4, 8),
    # up3_res: 96→96 k333, T=6 gives -19%
    (4, 32, 96, 96, (3, 3, 3), 184, 40): (96, 96, 6, 8, 4),
    # conv_out: 96→3 k333, T=9 gives -55%
    (4, 32, 96, 3, (3, 3, 3), 184, 40): (96, 32, 9, 8, 4),
    # --- BH Loud Box 2x4, simulated 4x32 Galaxy shapes (latent H=46, W=20) ---
    # Same per-device dims as 4x32 720p. Same optimized blockings.
    (2, 4, 32, 384, (3, 3, 3), 23, 5): (32, 128, 1, 16, 2),  # conv_in
    (2, 4, 384, 384, (3, 3, 3), 23, 5): (96, 96, 1, 32, 1),  # lat_res + mid_res
    (2, 4, 384, 768, (3, 1, 1), 23, 5): (128, 256, 3, 8, 4),  # up0_tconv
    (2, 4, 384, 192, (1, 3, 3), 46, 10): (96, 96, 1, 16, 4),  # up0_spatial
    (2, 4, 192, 384, (3, 3, 3), 46, 10): (96, 128, 3, 8, 4),  # up1_res0
    (2, 4, 384, 384, (3, 3, 3), 46, 10): (96, 128, 1, 16, 2),  # up1_res
    (2, 4, 384, 768, (3, 1, 1), 46, 10): (192, 384, 3, 16, 2),  # up1_tconv
    (2, 4, 384, 192, (1, 3, 3), 92, 20): (192, 96, 1, 32, 4),  # up1_spatial
    (2, 4, 192, 192, (3, 3, 3), 92, 20): (96, 96, 3, 8, 4),  # up2_res
    (2, 4, 192, 96, (1, 3, 3), 184, 40): (192, 96, 1, 4, 8),  # up2_spatial
    (2, 4, 96, 96, (3, 3, 3), 184, 40): (96, 96, 6, 8, 4),  # up3_res
    (2, 4, 96, 3, (3, 3, 3), 184, 40): (96, 32, 9, 8, 4),  # conv_out
    # --- BH Loud Box 2x4, 720p — v1: based on 4x8 720p with T_out_block > 1 ---
    # Per-device: lat(45,40) up0(90,80) up1(180,160) up2/up3(360,320)
    (2, 4, 32, 384, (3, 3, 3), 45, 40): (32, 128, 1, 16, 16),  # conv_in
    (2, 4, 384, 384, (3, 3, 3), 45, 40): (128, 128, 1, 4, 4),  # lat_res + mid_res
    (2, 4, 384, 768, (3, 1, 1), 45, 40): (192, 256, 3, 4, 4),  # up0_tconv (T=3)
    (2, 4, 384, 192, (1, 3, 3), 90, 80): (96, 96, 1, 16, 16),  # up0_spatial
    (2, 4, 192, 384, (3, 3, 3), 90, 80): (96, 128, 3, 8, 4),  # up1_res0 (T=3)
    (2, 4, 384, 384, (3, 3, 3), 90, 80): (128, 128, 1, 4, 4),  # up1_res
    (2, 4, 384, 768, (3, 1, 1), 90, 80): (192, 256, 3, 4, 4),  # up1_tconv (T=3)
    (2, 4, 384, 192, (1, 3, 3), 180, 160): (96, 96, 1, 16, 16),  # up1_spatial
    (2, 4, 192, 192, (3, 3, 3), 180, 160): (96, 96, 3, 4, 16),  # up2_res (T=3)
    (2, 4, 192, 96, (1, 3, 3), 360, 320): (192, 96, 1, 4, 32),  # up2_spatial
    (2, 4, 96, 96, (3, 3, 3), 360, 320): (96, 96, 3, 8, 8),  # up3_res (T=3)
    (2, 4, 96, 3, (3, 3, 3), 360, 320): (96, 32, 3, 8, 16),  # conv_out (T=3)
    # --- BH Galaxy / WH Galaxy 4x8, 720p — v3 (per-layer swept for T_latent=16/21) ---
    # h_dev=184, w_dev=160. Stages: lat(23,20) up0(46,40) up1(92,80) up2(184,160)
    # T values for T_latent=16: lat(17-18) up1(32-34) up2/up3(62-66) conv_out(62-66)
    (4, 8, 32, 384, (3, 3, 3), 23, 20): (32, 128, 1, 16, 16),  # conv_in (T=1 best)
    (4, 8, 384, 384, (3, 3, 3), 23, 20): (96, 96, 1, 4, 8),  # lat_res: H=4 W=8 -40% vs (128,128,1,4,4)
    (4, 8, 384, 768, (3, 1, 1), 23, 20): (192, 256, 2, 4, 4),  # up0_tconv: T=2 -33%
    (4, 8, 384, 192, (1, 3, 3), 46, 40): (96, 96, 1, 16, 16),  # up0_spatial (unchanged)
    (4, 8, 192, 384, (3, 3, 3), 46, 40): (96, 128, 2, 8, 4),  # up1_res0: T=2 -9%
    (4, 8, 384, 384, (3, 3, 3), 46, 40): (96, 128, 1, 8, 4),  # up1_res: Cin=96 H=8 -54% vs (128,128,1,4,4)
    (4, 8, 384, 768, (3, 1, 1), 46, 40): (192, 256, 2, 4, 4),  # up1_tconv: T=2 -7%
    (4, 8, 384, 192, (1, 3, 3), 92, 80): (96, 96, 1, 16, 16),  # up1_spatial (unchanged)
    (4, 8, 192, 192, (3, 3, 3), 92, 80): (96, 96, 4, 8, 4),  # up2_res: T=4 H=8 -30% vs (96,96,1,4,16)
    (4, 8, 192, 96, (1, 3, 3), 184, 160): (192, 96, 1, 4, 32),  # up2_spatial (unchanged)
    (4, 8, 96, 96, (3, 3, 3), 184, 160): (96, 96, 2, 8, 8),  # up3_res: T=2 -32% vs (96,96,1,8,8)
    (4, 8, 96, 3, (3, 3, 3), 184, 160): (96, 32, 33, 8, 16),  # conv_out: T=33 -60% vs (96,32,1,8,16)
    # --- BH Loud Box 2x4, 480p — v3: T_out_block > 1 for k333 layers ---
    # h_dev=240, w_dev=208. Stages: lat(30,26) up0(60,52) up1(120,104) up2(240,208)
    (2, 4, 32, 384, (3, 3, 3), 30, 26): (32, 128, 1, 16, 16),  # conv_in
    (2, 4, 384, 384, (3, 3, 3), 30, 26): (128, 128, 1, 8, 2),  # lat_res/mid_res
    (2, 4, 384, 768, (3, 1, 1), 30, 26): (128, 384, 3, 8, 2),  # up0_tconv (T=3)
    (2, 4, 384, 192, (1, 3, 3), 60, 52): (128, 96, 1, 16, 16),  # up0_spatial
    (2, 4, 192, 384, (3, 3, 3), 60, 52): (96, 128, 3, 4, 8),  # up1_res0 (T=3)
    (2, 4, 384, 384, (3, 3, 3), 60, 52): (128, 128, 1, 8, 2),  # up1_res
    (2, 4, 384, 768, (3, 1, 1), 60, 52): (128, 384, 3, 8, 2),  # up1_tconv (T=3)
    (2, 4, 384, 192, (1, 3, 3), 120, 104): (128, 96, 1, 16, 16),  # up1_spatial
    (2, 4, 192, 192, (3, 3, 3), 120, 104): (96, 96, 3, 8, 8),  # up2_res (T=3, already optimal)
    (2, 4, 192, 96, (1, 3, 3), 240, 208): (192, 96, 1, 16, 8),  # up2_spatial
    (2, 4, 96, 96, (3, 3, 3), 240, 208): (96, 96, 3, 4, 16),  # up3_res (T=3 H=4 W=16, -4%)
    (2, 4, 96, 3, (3, 3, 3), 240, 208): (96, 32, 6, 16, 8),  # conv_out (T=6 H=16 W=8, -29%)
}

# Fallback table: (C_in, C_out, kernel) -> blocking.
# Used when no exact (mesh, spatial) match exists.
# MUST match main branch blockings — this is the safe fallback path.
_DEFAULT_BLOCKINGS = {
    (96, 3, (3, 3, 3)): (96, 32, 1, 16, 8),
    (96, 32, (3, 3, 3)): (96, 32, 1, 16, 8),
    (192, 96, (1, 3, 3)): (192, 96, 1, 4, 8),
    (96, 96, (3, 3, 3)): (96, 96, 1, 8, 8),
    (384, 192, (1, 3, 3)): (192, 96, 1, 32, 4),
    (192, 192, (3, 3, 3)): (96, 96, 1, 8, 4),
    (32, 384, (3, 3, 3)): (32, 96, 1, 2, 32),
    (192, 384, (3, 3, 3)): (64, 128, 1, 8, 4),
    (384, 384, (3, 3, 3)): (96, 96, 1, 8, 4),
    (384, 768, (3, 3, 3)): (96, 96, 1, 8, 4),
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
    key = (h_factor, w_factor, in_channels, out_channels, kernel_size, H_out, W_out)
    logger.debug(f"get_conv3d_config key={key} dtype={weights_dtype}")
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
