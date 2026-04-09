# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import collections
import hashlib
import math
from itertools import repeat
from typing import NamedTuple

import torch
from loguru import logger

import ttnn

from ..layers.module import Module

ALIGNMENT = 32


def aligned_channels(channels):
    ALIGN_PAD = ALIGNMENT - channels % ALIGNMENT
    if channels % ALIGNMENT != 0:
        channels = channels + ALIGN_PAD
    return channels


def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == n, f"{x} must be a tuple of length {n}"
        return tuple(x)
    return tuple(repeat(x, n))


class StageHW(NamedTuple):
    H: int
    W: int


class StageT(NamedTuple):
    T_res: int  # T for residual blocks (cur_T + 2 for k333 causal padding)
    T_tconv: int  # T for temporal conv (cur_T + 1), or 0 if no temporal upsample
    T_spatial: int  # T after temporal upsample (for spatial conv), equals cur_T if none


def compute_decoder_dims(
    target_height, target_width, h_factor, w_factor, t_chunk_size, *, temperal_upsample, num_stages=3
):
    """Compute per-stage spatial and temporal dimensions for a VAE decoder.

    Returns (stage_hw, stage_t) where:
      stage_hw: list of StageHW per stage (length num_stages + 1), latent -> full resolution
      stage_t:  list of StageT per stage (length num_stages + 1)
    """
    vae_scale = 2**num_stages
    lat_h = math.ceil(target_height / vae_scale / h_factor)
    lat_w = target_width // vae_scale // w_factor
    stage_hw = [StageHW(lat_h * (2**s), lat_w * (2**s)) for s in range(num_stages + 1)]

    cur_T = t_chunk_size
    stage_t = []
    for i in range(num_stages + 1):
        has_temporal_up = i < len(temperal_upsample) and temperal_upsample[i]
        T_spatial = (2 * (cur_T - 1) + 1) if has_temporal_up else cur_T
        stage_t.append(
            StageT(
                T_res=cur_T + 2,
                T_tconv=cur_T + 1 if has_temporal_up else 0,
                T_spatial=T_spatial,
            )
        )
        if has_temporal_up:
            cur_T = T_spatial

    return stage_hw, stage_t


# Blocking table: (h_factor, w_factor, C_in, C_out, kernel, T, H, W) -> blocking
# Each production (mesh, resolution, temporal-mode, layer) combination gets its own entry.
# Blockings are (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).
_BLOCKINGS = {
    # ===================================================================
    # BH Galaxy 6U 4x32, 720p, 81 frames full-T (latent T=21)
    # Per-device: lat(23,5) mid(46,10) hi(92,20) full(184,40)
    # ===================================================================
    (4, 32, 32, 384, (3, 3, 3), 23, 23, 5): (32, 128, 1, 16, 2),  # conv_in
    (4, 32, 384, 384, (3, 3, 3), 23, 23, 5): (96, 96, 3, 32, 1),  # lat_res+mid_res
    (4, 32, 384, 768, (3, 1, 1), 22, 23, 5): (128, 256, 3, 8, 4),  # up0_tconv
    (4, 32, 384, 192, (1, 3, 3), 41, 46, 10): (96, 96, 1, 16, 4),  # up0_spatial
    (4, 32, 192, 384, (3, 3, 3), 43, 46, 10): (96, 128, 3, 8, 4),  # up1_res0
    (4, 32, 384, 384, (3, 3, 3), 43, 46, 10): (96, 128, 3, 16, 2),  # up1_res
    (4, 32, 384, 768, (3, 1, 1), 42, 46, 10): (192, 384, 3, 16, 2),  # up1_tconv
    (4, 32, 384, 192, (1, 3, 3), 81, 92, 20): (192, 96, 1, 32, 4),  # up1_spatial
    (4, 32, 192, 192, (3, 3, 3), 83, 92, 20): (96, 96, 9, 8, 4),  # up2_res
    (4, 32, 192, 96, (1, 3, 3), 81, 184, 40): (192, 96, 1, 4, 8),  # up2_spatial
    (4, 32, 96, 96, (3, 3, 3), 83, 184, 40): (96, 96, 6, 8, 4),  # up3_res
    (4, 32, 96, 3, (3, 3, 3), 83, 184, 40): (96, 32, 9, 8, 4),  # conv_out
    # ===================================================================
    # BH Galaxy 4x8, 720p, 81 frames full-T (latent T=21)
    # Per-device: lat(23,20) mid(46,40) hi(92,80) full(184,160)
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 23, 23, 20): (32, 128, 1, 16, 16),  # conv_in
    (4, 8, 384, 384, (3, 3, 3), 23, 23, 20): (96, 96, 3, 8, 4),  # lat_res+mid_res
    (4, 8, 384, 768, (3, 1, 1), 22, 23, 20): (192, 256, 1, 4, 4),  # up0_tconv
    (4, 8, 384, 192, (1, 3, 3), 41, 46, 40): (96, 96, 1, 16, 16),  # up0_spatial
    (4, 8, 192, 384, (3, 3, 3), 43, 46, 40): (96, 128, 3, 8, 4),  # up1_res0
    (4, 8, 384, 384, (3, 3, 3), 43, 46, 40): (96, 96, 3, 8, 8),  # up1_res
    (4, 8, 384, 768, (3, 1, 1), 42, 46, 40): (192, 256, 3, 4, 4),  # up1_tconv
    (4, 8, 384, 192, (1, 3, 3), 81, 92, 80): (96, 96, 1, 16, 16),  # up1_spatial
    (4, 8, 192, 192, (3, 3, 3), 83, 92, 80): (96, 96, 3, 8, 8),  # up2_res
    (4, 8, 192, 96, (1, 3, 3), 81, 184, 160): (192, 96, 1, 4, 32),  # up2_spatial
    (4, 8, 96, 96, (3, 3, 3), 83, 184, 160): (96, 96, 5, 8, 8),  # up3_res
    (4, 8, 96, 3, (3, 3, 3), 83, 184, 160): (96, 32, 3, 8, 16),  # conv_out
    # ===================================================================
    # BH Galaxy 4x8, 720p, 81 frames t_chunk_size=11 (latent T=11)
    # Per-device: same spatial as above
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 13, 23, 20): (32, 128, 2, 16, 16),  # conv_in
    (4, 8, 384, 384, (3, 3, 3), 13, 23, 20): (96, 96, 1, 4, 8),  # lat_res+mid_res
    (4, 8, 384, 768, (3, 1, 1), 12, 23, 20): (192, 256, 2, 4, 4),  # up0_tconv
    (4, 8, 384, 192, (1, 3, 3), 21, 46, 40): (96, 96, 1, 16, 16),  # up0_spatial
    (4, 8, 192, 384, (3, 3, 3), 23, 46, 40): (96, 128, 2, 8, 4),  # up1_res0
    (4, 8, 384, 384, (3, 3, 3), 23, 46, 40): (96, 128, 2, 8, 4),  # up1_res
    (4, 8, 384, 768, (3, 1, 1), 22, 46, 40): (192, 256, 2, 4, 4),  # up1_tconv
    (4, 8, 384, 192, (1, 3, 3), 41, 92, 80): (96, 96, 1, 16, 16),  # up1_spatial
    (4, 8, 192, 192, (3, 3, 3), 43, 92, 80): (96, 96, 4, 8, 4),  # up2_res
    (4, 8, 192, 96, (1, 3, 3), 41, 184, 160): (192, 96, 1, 4, 32),  # up2_spatial
    (4, 8, 96, 96, (3, 3, 3), 43, 184, 160): (96, 96, 4, 8, 4),  # up3_res
    (4, 8, 96, 3, (3, 3, 3), 43, 184, 160): (96, 32, 31, 4, 8),  # conv_out
    # ===================================================================
    # BH Loud Box 4x2, 480p, 81 frames cached (t_chunk_size=1)
    # Per-device: lat(15,52) mid(30,104) hi(60,208) full(120,416)
    # ===================================================================
    (4, 2, 32, 384, (3, 3, 3), 3, 15, 52): (32, 128, 1, 16, 16),  # conv_in
    (4, 2, 384, 384, (3, 3, 3), 3, 15, 52): (128, 128, 1, 8, 2),  # lat_res+mid_res
    (4, 2, 384, 768, (3, 1, 1), 2, 15, 52): (128, 384, 3, 8, 2),  # up0_tconv
    (4, 2, 384, 192, (1, 3, 3), 1, 30, 104): (128, 96, 1, 16, 16),  # up0_spatial
    (4, 2, 192, 384, (3, 3, 3), 3, 30, 104): (96, 128, 3, 4, 8),  # up1_res0
    (4, 2, 384, 384, (3, 3, 3), 3, 30, 104): (128, 128, 1, 8, 2),  # up1_res
    (4, 2, 384, 768, (3, 1, 1), 2, 30, 104): (128, 384, 3, 8, 2),  # up1_tconv
    (4, 2, 384, 192, (1, 3, 3), 1, 60, 208): (128, 96, 1, 16, 16),  # up1_spatial
    (4, 2, 192, 192, (3, 3, 3), 3, 60, 208): (96, 96, 3, 8, 8),  # up2_res
    (4, 2, 192, 96, (1, 3, 3), 1, 120, 416): (192, 96, 1, 16, 8),  # up2_spatial
    (4, 2, 96, 96, (3, 3, 3), 3, 120, 416): (96, 96, 3, 4, 16),  # up3_res
    (4, 2, 96, 3, (3, 3, 3), 3, 120, 416): (96, 32, 6, 16, 8),  # conv_out
}

# Fallback table: (C_in, C_out, kernel) -> blocking.
# Used when no exact (mesh, spatial) match exists.
# MUST match main branch blockings -- this is the safe fallback path.
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
    in_channels, out_channels, kernel_size, weights_dtype, grid_size, *, h_factor=1, w_factor=1, T=0, H=0, W=0
):
    """Get optimized Conv3dConfig for a conv3d layer.

    Lookup chain: exact (mesh, shape, T, spatial) match -> fallback (channel, kernel) match -> default.
    Pass h_factor, w_factor, T, H, W for best results. When these are not
    available the fallback table is used.
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

    key = (h_factor, w_factor, in_channels, out_channels, kernel_size, T, H, W)
    shape_key = (in_channels, out_channels, kernel_size)

    exact = _BLOCKINGS.get(key)
    if exact is not None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = exact
        logger.info(
            f"conv3d blocking [exact] {key} -> "
            f"Cin={C_in_block} Cout={C_out_block} T={T_out_block} H={H_out_block} W={W_out_block}"
        )
    else:
        fallback = _DEFAULT_BLOCKINGS.get(shape_key)
        if fallback is not None:
            C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = fallback
            logger.warning(
                f"conv3d blocking [fallback] {key} -> shape_key={shape_key} -> "
                f"Cin={C_in_block} Cout={C_out_block} T={T_out_block} H={H_out_block} W={W_out_block}"
            )
        else:
            C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = in_channels, 32, 1, 1, 1
            logger.warning(
                f"conv3d blocking [NONE] {key} -> no match in any table, using hardcoded default: "
                f"Cin={C_in_block} Cout={C_out_block} T={T_out_block} H={H_out_block} W={W_out_block}"
            )

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
    pad_h = (h_factor - H % h_factor) % h_factor
    if pad_h > 0:
        tensor_BTHWC = torch.nn.functional.pad(tensor_BTHWC, (0, 0, 0, 0, 0, pad_h))
    return tensor_BTHWC, H


def conv_unpad_height(tensor_BTHWC, logical_h):
    """
    For Wan2.2, remove height padding that was added by conv_pad_height.
    """
    B, T, H, W, C = tensor_BTHWC.shape
    return tensor_BTHWC[:, :, :logical_h, :, :]


def conv_pad_in_channels(tensor):
    C_in = tensor.shape[-1]
    padded_C_in = aligned_channels(C_in)
    if padded_C_in != C_in:
        tensor = torch.nn.functional.pad(tensor, (0, padded_C_in - C_in))
    return tensor
