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


def aligned_channels(channels, unit: int = ALIGNMENT):
    """Round ``channels`` up to a multiple of ``unit`` (default TILE_WIDTH).

    Channel-TP passes ``unit = factor * TILE_WIDTH`` so each per-chip shard is
    itself a TILE_WIDTH multiple (a 48-wide shard is illegal in TILE layout).
    """
    rem = channels % unit
    if rem != 0:
        channels = channels + (unit - rem)
    return channels


def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == n, f"{x} must be a tuple of length {n}"
        return tuple(x)
    return tuple(repeat(x, n))


class ConvDims(NamedTuple):
    """Target (T, H, W) dimensions for a single conv3d layer, used for blocking lookup."""

    T: int = 0
    H: int = 0
    W: int = 0


class StageHW(NamedTuple):
    H: int
    W: int


class StageT(NamedTuple):
    T_res: int  # T seen by residual-block conv3d = cur_T + 2 (causal pad)
    T_tconv: int  # T seen by time_conv conv3d, or 0 if no temporal upsample
    T_spatial: int  # T after temporal upsample (input to spatial conv), equals cur_T if none


def compute_decoder_dims(
    height, width, h_factor, w_factor, t_chunk_size, *, temperal_upsample, num_stages=3, cached=False
):
    """Compute per-stage spatial and temporal dimensions for a VAE decoder.

    Returns (stage_hw, stage_t) where:
      stage_hw: list of StageHW per stage (length num_stages + 1), latent -> full resolution
      stage_t:  list of StageT per stage (length num_stages + 1)

    t_chunk_size must be a concrete integer — the latent T that will be processed:
      full-T:   pass the full latent frame count (e.g. 21), with cached=False
      chunked:  pass the chunk size (e.g. 1, 7), with cached=True
    Returns zero dims when t_chunk_size is None or < 1 (constructor default fallback).

    Temporal formulas differ between uncached and cached paths because
    WanResample splits off frame-0 in the uncached path but not in the cached path:

      uncached: T_tconv   = cur_T + 1           (frames[1:] = cur_T-1, +2 causal pad)
                T_spatial = 2*(cur_T-1) + 1     (frame-0 + doubled rest)
      cached:   T_tconv   = cur_T + 2           (all frames + 2 causal pad from cache)
                T_spatial = 2 * cur_T           (all frames doubled)
    """
    if t_chunk_size is None or t_chunk_size < 1:
        n = num_stages + 1
        return [StageHW(0, 0)] * n, [StageT(T_res=0, T_tconv=0, T_spatial=0)] * n

    vae_scale = 2**num_stages
    # Both height and width use ceil because some configs don't divide evenly
    # (e.g. 720/8/4=22.5 → 23).  The hardware pads via conv_pad_height / conv_pad_width.
    lat_h = math.ceil(height / vae_scale / h_factor)
    lat_w = math.ceil(width / vae_scale / w_factor)
    stage_hw = [StageHW(lat_h * (2**s), lat_w * (2**s)) for s in range(num_stages + 1)]

    cur_T = t_chunk_size
    stage_t = []
    for i in range(num_stages + 1):
        has_temporal_up = i < len(temperal_upsample) and temperal_upsample[i]
        if cached:
            T_tconv = (cur_T + 2) if has_temporal_up else 0
            T_spatial = (2 * cur_T) if has_temporal_up else cur_T
        else:
            T_tconv = (cur_T + 1) if has_temporal_up else 0
            T_spatial = (2 * (cur_T - 1) + 1) if has_temporal_up else cur_T
        stage_t.append(StageT(T_res=cur_T + 2, T_tconv=T_tconv, T_spatial=T_spatial))
        if has_temporal_up:
            cur_T = T_spatial

    return stage_hw, stage_t


def compute_encoder_dims(height, width, h_factor, w_factor, encoder_t_chunk_size, *, temperal_downsample, num_stages=3):
    """Compute per-stage spatial and temporal dimensions for a cached VAE encoder.

    Returns (stage_hw, stage_t) where:
      stage_hw: list of StageHW per stage (length num_stages + 1), full → latent resolution
      stage_t:  list of StageT per stage (length num_stages + 1)

    encoder_t_chunk_size must be a concrete integer — the pixel-frame count being encoded
    (e.g. _I2V_ENCODE_FRAMES=33). Returns zero dims when None or < 1 (constructor default fallback).

    Downsample order: spatial WanConv2d at current res → 2x2 slice → WanCausalConv3d time_conv at halved res.
    T_res     = cur_T + 2  (3D causal convs: residual blocks, conv_in, conv_out)
    T_spatial = cur_T      (WanConv2d (1,3,3): no causal cache needed, sees exactly cur_T frames)
    T_tconv   = cur_T      (time_conv (3,1,1) at next spatial res: input from spatial output = cur_T frames)
    after_down = (cur_T - 1) // 2 + 1  (strided temporal conv output → next stage cur_T)
    """
    n = num_stages + 1
    if encoder_t_chunk_size is None or encoder_t_chunk_size < 1:
        return [StageHW(0, 0)] * n, [StageT(T_res=0, T_tconv=0, T_spatial=0)] * n

    full_h = math.ceil(height / h_factor)
    full_w = math.ceil(width / w_factor)
    stage_hw = [StageHW(full_h // (2**s), full_w // (2**s)) for s in range(n)]

    cur_T = encoder_t_chunk_size
    stage_t = []
    for i in range(n):
        has_temporal_down = i < len(temperal_downsample) and temperal_downsample[i]
        T_tconv = cur_T if has_temporal_down else 0
        stage_t.append(StageT(T_res=cur_T + 2, T_tconv=T_tconv, T_spatial=cur_T))
        if has_temporal_down:
            cur_T = (cur_T - 1) // 2 + 1

    return stage_hw, stage_t


# Blocking table: (h_factor, w_factor, C_in, C_out, kernel, T, H, W) -> blocking
# Values updated 2026-06 from a HiFi2 trace-timed re-sweep; inline us are HiFi2
# per-op times for re-swept entries. See bruteforce_conv3d_sweep.py.
# Each production (mesh, resolution, temporal-mode, layer) combination gets its own entry.
# Blockings are (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).
_BLOCKINGS = {
    # ===================================================================
    # BH Galaxy 6U 4x32, 480p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(15,3) stage1(30,6) stage2(60,12) stage3(120,24)
    # Padded (int_pad=(0,1,1)): stage0(17,5) stage1(32,8) stage2(62,14) stage3(122,26)
    # Swept 2026-04-13 on 1x1 mesh; results in sweep_results_h4w32_480p_full_t/.
    # hw_product=32 + max_t_block=8. T=3-7 wins. Large-C_in tconv/res partial.
    # ===================================================================
    (4, 32, 32, 384, (3, 3, 3), 23, 15, 3): (32, 128, 1, 8, 4),  # conv_in — 42us
    (4, 32, 384, 384, (3, 3, 3), 23, 15, 3): (128, 64, 7, 8, 4),  # lat_mid_res — 180us
    (4, 32, 384, 768, (3, 1, 1), 22, 15, 3): (192, 256, 1, 8, 4),  # up0_tconv — 240us table
    (4, 32, 384, 192, (1, 3, 3), 41, 30, 6): (384, 96, 1, 16, 2),  # up0_spatial — 124us
    (4, 32, 192, 384, (3, 3, 3), 43, 30, 6): (96, 128, 5, 8, 4),  # up1_res0 — 410us
    (4, 32, 384, 384, (3, 3, 3), 43, 30, 6): (96, 128, 5, 4, 8),  # up1_res — 743us partial
    (4, 32, 384, 768, (3, 1, 1), 42, 30, 6): (192, 768, 1, 16, 2),  # up1_tconv — 150us partial
    (4, 32, 384, 192, (1, 3, 3), 81, 60, 12): (384, 96, 1, 8, 4),  # up1_spatial — 625us
    (4, 32, 192, 192, (3, 3, 3), 83, 60, 12): (96, 96, 7, 16, 2),  # up2_res — 1158us
    (4, 32, 192, 96, (1, 3, 3), 81, 120, 24): (192, 96, 1, 8, 4),  # up2_spatial — 1398us table
    (4, 32, 96, 96, (3, 3, 3), 83, 120, 24): (96, 96, 7, 16, 2),  # up3_res — 1173us
    (4, 32, 96, 3, (3, 3, 3), 83, 120, 24): (96, 32, 7, 2, 16),  # conv_out — 1161us
    # ===================================================================
    # BH Galaxy 6U 4x32, 720p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(23,5) stage1(46,10) stage2(92,20) stage3(184,40)
    # ===================================================================
    (4, 32, 32, 384, (3, 3, 3), 23, 23, 5): (32, 384, 1, 8, 4),  # conv_in
    (4, 32, 384, 384, (3, 3, 3), 23, 23, 5): (96, 96, 7, 16, 2),  # lat_res+mid_res
    (4, 32, 384, 768, (3, 1, 1), 22, 23, 5): (384, 256, 2, 8, 4),  # up0_tconv
    (4, 32, 384, 192, (1, 3, 3), 41, 46, 10): (384, 96, 1, 16, 2),  # up0_spatial
    (4, 32, 192, 384, (3, 3, 3), 43, 46, 10): (96, 128, 5, 8, 4),  # up1_res0
    (4, 32, 384, 384, (3, 3, 3), 43, 46, 10): (96, 128, 5, 16, 2),  # up1_res
    (4, 32, 384, 768, (3, 1, 1), 42, 46, 10): (384, 384, 2, 8, 4),  # up1_tconv
    (4, 32, 384, 192, (1, 3, 3), 81, 92, 20): (384, 96, 1, 8, 4),  # up1_spatial
    (4, 32, 192, 192, (3, 3, 3), 83, 92, 20): (96, 96, 9, 8, 4),  # up2_res
    (4, 32, 192, 96, (1, 3, 3), 81, 184, 40): (192, 96, 1, 8, 4),  # up2_spatial
    (4, 32, 96, 96, (3, 3, 3), 83, 184, 40): (96, 96, 6, 8, 4),  # up3_res
    (4, 32, 96, 3, (3, 3, 3), 83, 184, 40): (96, 32, 9, 8, 4),  # conv_out
    # ===================================================================
    # BH Galaxy 4x8, 480p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(15,13) stage1(30,26) stage2(60,52) stage3(120,104)
    # Padded (int_pad=(0,1,1)): stage0(17,15) stage1(32,28) stage2(62,54) stage3(122,106)
    # Swept 2026-04-10 on 1x1 mesh; results in sweep_results_h4w8_480p_full_t/.
    # hw_product=32 + max_t_block=8. Partial: large-C_in/tconv layers hung mid-sweep.
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 23, 15, 13): (32, 384, 1, 16, 2),  # conv_in — 62us
    (4, 8, 384, 384, (3, 3, 3), 23, 15, 13): (128, 96, 3, 16, 2),  # lat_mid_res — 407us
    (4, 8, 384, 768, (3, 1, 1), 22, 15, 13): (384, 384, 2, 8, 4),  # up0_tconv — 98us partial
    (4, 8, 384, 192, (1, 3, 3), 41, 30, 26): (384, 96, 1, 16, 2),  # up0_spatial — 344us
    (4, 8, 192, 384, (3, 3, 3), 43, 30, 26): (96, 128, 5, 8, 4),  # up1_res0 — 1032us partial
    (4, 8, 384, 384, (3, 3, 3), 43, 30, 26): (96, 128, 5, 8, 4),  # up1_res — 2209us partial
    (4, 8, 384, 768, (3, 1, 1), 42, 30, 26): (384, 384, 2, 8, 4),  # up1_tconv — 454us partial
    (4, 8, 384, 192, (1, 3, 3), 81, 60, 52): (384, 96, 1, 8, 4),  # up1_spatial — 2510us
    (4, 8, 192, 192, (3, 3, 3), 83, 60, 52): (96, 96, 7, 16, 2),  # up2_res — 4423us
    (4, 8, 192, 96, (1, 3, 3), 81, 120, 104): (192, 96, 1, 2, 16),  # up2_spatial — table 3091us
    (4, 8, 96, 96, (3, 3, 3), 83, 120, 104): (96, 96, 7, 2, 16),  # up3_res — 4482us
    (4, 8, 96, 3, (3, 3, 3), 83, 120, 104): (96, 32, 7, 2, 16),  # conv_out — 4378us
    # ===================================================================
    # BH Galaxy 4x8, 720p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(23,20) stage1(46,40) stage2(92,80) stage3(184,160)
    # Swept 2026-04-13 on 1x1 mesh; results in sweep_results_h4w8_720p_full_t/.
    # hw_product=32 + max_t_block=8. T=3-6 wins. tconv layers partial.
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 23, 23, 20): (32, 384, 3, 8, 4),  # conv_in — 96us
    (4, 8, 384, 384, (3, 3, 3), 23, 23, 20): (128, 96, 3, 8, 4),  # lat_mid_res — 762us table
    (4, 8, 384, 768, (3, 1, 1), 22, 23, 20): (384, 384, 2, 8, 4),  # up0_tconv — 182us partial
    (4, 8, 384, 192, (1, 3, 3), 41, 46, 40): (384, 96, 1, 8, 4),  # up0_spatial — 760us
    (4, 8, 192, 384, (3, 3, 3), 43, 46, 40): (96, 128, 5, 8, 4),  # up1_res0 — 2176us
    (4, 8, 384, 384, (3, 3, 3), 43, 46, 40): (96, 128, 5, 16, 2),  # up1_res — 4546us
    (4, 8, 384, 768, (3, 1, 1), 42, 46, 40): (384, 384, 3, 4, 8),  # up1_tconv — 888us partial
    (4, 8, 384, 192, (1, 3, 3), 81, 92, 80): (384, 96, 1, 4, 8),  # up1_spatial — 5490us
    (4, 8, 192, 192, (3, 3, 3), 83, 92, 80): (96, 96, 7, 16, 2),  # up2_res — 10833us
    (4, 8, 192, 96, (1, 3, 3), 81, 184, 160): (192, 96, 1, 2, 16),  # up2_spatial — 7101us table
    (4, 8, 96, 96, (3, 3, 3), 83, 184, 160): (96, 96, 4, 4, 16),  # up3_res — 16056us (hw=64)
    (4, 8, 96, 3, (3, 3, 3), 83, 184, 160): (96, 32, 3, 2, 16),  # conv_out — 10094us
    # ===================================================================
    # BH Galaxy 4x8, 720p, cached t_chunk_size=1 (vae_t_chunk_size=1)
    # First chunk / anchor frame: all stages see T = t_chunk + 2 = 3.
    # Swept 2026-04-11 on 1x1 mesh; results in sweep_results_h4w8_720p_t1/.
    # T_out=1 for (3,3,3) → only T_block=1 tested. Large-C_in tconv layers partial.
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 3, 23, 20): (32, 128, 1, 8, 4),  # conv_in — 28us
    (4, 8, 384, 384, (3, 3, 3), 3, 23, 20): (128, 64, 1, 4, 8),  # lat_mid_res — 137us
    (4, 8, 384, 768, (3, 1, 1), 3, 23, 20): (192, 384, 1, 8, 4),  # up0_tconv — 34us partial
    (4, 8, 384, 192, (1, 3, 3), 3, 46, 40): (384, 96, 1, 8, 4),  # up0_spatial — 80us
    (4, 8, 192, 384, (3, 3, 3), 3, 46, 40): (64, 128, 1, 16, 2),  # up1_res0 — 157us
    (4, 8, 384, 384, (3, 3, 3), 3, 46, 40): (128, 96, 1, 16, 2),  # up1_res — 257us
    (4, 8, 384, 768, (3, 1, 1), 3, 46, 40): (384, 384, 1, 4, 8),  # up1_tconv — 67us partial
    (4, 8, 384, 192, (1, 3, 3), 3, 92, 80): (384, 96, 1, 16, 2),  # up1_spatial — 203us
    (4, 8, 192, 192, (3, 3, 3), 3, 92, 80): (96, 96, 1, 16, 2),  # up2_res — 455us
    (4, 8, 192, 96, (1, 3, 3), 3, 184, 160): (192, 96, 1, 16, 2),  # up2_spatial — 615us
    (4, 8, 96, 96, (3, 3, 3), 3, 184, 160): (96, 96, 1, 2, 16),  # up3_res — table 238us
    (4, 8, 96, 3, (3, 3, 3), 3, 184, 160): (96, 32, 1, 2, 16),  # conv_out — 227us
    # ===================================================================
    # BH Galaxy 4x8, 720p, cached t_chunk_size=15 (vae_t_chunk_size=15)
    # Stage 0: T_res=17, T_tconv=17. Stage 1: T_res=32, T_tconv=32, T_sp=60.
    # Stage 2/3: T_res=62. Spatial bridge T: T_sp_stage0=30, T_sp_stage1=60.
    # Swept 2026-04-11 on 1x1 mesh; results in sweep_results_h4w8_720p_t15/.
    # hw_product=32 + max_t_block=8. T=3-7 win. Large-C_in tconv layers partial.
    # ===================================================================
    # Stage 0 (cur_T=15)
    (4, 8, 32, 384, (3, 3, 3), 17, 23, 20): (32, 384, 3, 8, 4),  # conv_in — 84us
    (4, 8, 384, 384, (3, 3, 3), 17, 23, 20): (96, 128, 5, 8, 4),  # lat_mid_res — 490us partial
    (4, 8, 384, 768, (3, 1, 1), 17, 23, 20): (192, 768, 1, 8, 4),  # up0_tconv — 138us partial
    (4, 8, 384, 192, (1, 3, 3), 30, 46, 40): (384, 64, 1, 8, 4),  # up0_spatial — 636us
    # Stage 1 (cur_T=30)
    (4, 8, 192, 384, (3, 3, 3), 32, 46, 40): (96, 128, 5, 16, 2),  # up1_res0 — 1451us
    (4, 8, 384, 384, (3, 3, 3), 32, 46, 40): (128, 96, 2, 16, 2),  # up1_res — 3721us
    (4, 8, 384, 768, (3, 1, 1), 32, 46, 40): (384, 384, 3, 4, 8),  # up1_tconv — 672us partial
    (4, 8, 384, 192, (1, 3, 3), 60, 92, 80): (384, 64, 1, 16, 2),  # up1_spatial — 4617us
    # Stage 2/3 (cur_T=60, no temporal upsample)
    (4, 8, 192, 192, (3, 3, 3), 62, 92, 80): (96, 96, 7, 16, 2),  # up2_res — 7601us
    (4, 8, 192, 96, (1, 3, 3), 60, 184, 160): (192, 96, 1, 2, 16),  # up2_spatial — 6067us
    (4, 8, 96, 96, (3, 3, 3), 62, 184, 160): (96, 96, 7, 2, 16),  # up3_res — 12020us
    (4, 8, 96, 3, (3, 3, 3), 62, 184, 160): (96, 32, 7, 8, 4),  # conv_out — 7208us
    # ===================================================================
    # BH Galaxy 4x8, 720p, cached t_chunk_size=16 (vae_t_chunk_size=16)
    # BH (4,8): h_factor=4, w_factor=8. Per-device (H,W): same spatial as full-T above.
    # Padded (int_pad=(0,1,1)): stage0(25,22) stage1(48,42) stage2(94,82) stage3(186,162)
    # Cached T: stage0(T_res=18,T_tconv=18,T_sp=32) stage1(34,34,64) stage2/3(66,_,64)
    # Swept 2026-04-10 on 1x1 mesh; results in sweep_results_h4w8_720p_t16/.
    # T=8 (t_chunk/2) wins for large stages; T=2-4 for mid; T=1 for tconv.
    # hw_product=32 prevents device hangs. max_t_block=8 (T=9+ hangs).
    # Partial: large-C_in layers (384ch) hung before sweep completion.
    # ===================================================================
    # Stage 0 (cur_T=16)
    (4, 8, 32, 384, (3, 3, 3), 18, 23, 20): (32, 384, 2, 2, 16),  # conv_in — 82us
    (4, 8, 384, 384, (3, 3, 3), 18, 23, 20): (96, 128, 2, 8, 4),  # lat_mid_res — 598us partial
    (4, 8, 384, 768, (3, 1, 1), 18, 23, 20): (192, 768, 1, 8, 4),  # up0_tconv — 144us partial
    (4, 8, 384, 192, (1, 3, 3), 32, 46, 40): (384, 64, 1, 16, 2),  # up0_spatial — 664us
    # Stage 1 (cur_T=32)
    (4, 8, 192, 384, (3, 3, 3), 34, 46, 40): (96, 128, 5, 8, 4),  # up1_res0 — 1947us partial
    (4, 8, 384, 384, (3, 3, 3), 34, 46, 40): (96, 128, 4, 16, 2),  # up1_res — 3532us partial
    (4, 8, 384, 768, (3, 1, 1), 34, 46, 40): (384, 384, 3, 4, 8),  # up1_tconv — 749us partial
    (4, 8, 384, 192, (1, 3, 3), 64, 92, 80): (384, 64, 1, 16, 2),  # up1_spatial — 4711us
    # Stage 2 (cur_T=64, no temporal upsample)
    (4, 8, 192, 192, (3, 3, 3), 66, 92, 80): (96, 96, 6, 8, 4),  # up2_res — 8746us
    (4, 8, 192, 96, (1, 3, 3), 64, 184, 160): (192, 96, 1, 2, 16),  # up2_spatial — 6077us
    # Stage 3 (cur_T=64, no temporal upsample)
    (4, 8, 96, 96, (3, 3, 3), 66, 184, 160): (96, 96, 8, 8, 4),  # up3_res — 7730us
    (4, 8, 96, 3, (3, 3, 3), 66, 184, 160): (96, 32, 5, 8, 4),  # conv_out — 7972us
    # ===================================================================
    # BH Loud Box 2x4, 480p, cached t_chunk_size=7 (vae_t_chunk_size=7)
    # BH (2,4): tp_axis=0, sp_axis=1 → h_factor=2, w_factor=4
    # Per-device (H,W): stage0(30,26) stage1(60,52) stage2(120,104) stage3(240,208)
    # Cached T: cur_T grows 7 → 14 → 28 across stages
    # Swept 2026-04-10 on BH Loud Box 2x4; results stored in sweep_results_h2w4_480p_t7/
    # Note: lat_mid_res, up0_tconv, up1_res0/res, up2_res, up3_res, conv_out are
    # partial sweeps (device hangs after first T>1 combos — see CONV3D_BLOCKING_SWEEP_BH2X4_480P.md).
    # ===================================================================
    # Stage 0 (cur_T=7): T_res=9, T_tconv=9, T_spatial=14
    (2, 4, 32, 384, (3, 3, 3), 9, 30, 26): (32, 384, 1, 2, 16),  # conv_in — swept 65us
    (2, 4, 384, 384, (3, 3, 3), 9, 30, 26): (96, 96, 7, 16, 2),  # lat_mid_res — partial 535us
    (2, 4, 384, 768, (3, 1, 1), 9, 30, 26): (192, 768, 1, 16, 2),  # up0_tconv — partial 126us
    (2, 4, 384, 192, (1, 3, 3), 14, 60, 52): (384, 96, 1, 4, 8),  # up0_spatial — table wins 459us
    # Stage 1 (cur_T=14): T_res=16, T_tconv=16, T_spatial=28
    (2, 4, 192, 384, (3, 3, 3), 16, 60, 52): (96, 128, 5, 4, 8),  # up1_res0 — partial 1460us
    (2, 4, 384, 384, (3, 3, 3), 16, 60, 52): (96, 128, 5, 8, 4),  # up1_res — inferred from up1_res0
    (2, 4, 384, 768, (3, 1, 1), 16, 60, 52): (384, 384, 3, 8, 4),  # up1_tconv — swept 591us
    (2, 4, 384, 192, (1, 3, 3), 28, 120, 104): (384, 96, 1, 4, 8),  # up1_spatial — swept 6809us
    # Stage 2 (cur_T=28): T_res=30, T_spatial=28 (no temporal upsample)
    (2, 4, 192, 192, (3, 3, 3), 30, 120, 104): (96, 96, 6, 8, 4),  # up2_res — swept 6028us
    (2, 4, 192, 96, (1, 3, 3), 28, 240, 208): (192, 96, 1, 4, 16),  # up2_spatial — swept 6509us
    # Stage 3 (cur_T=28): T_res=30 (no temporal upsample)
    (2, 4, 96, 96, (3, 3, 3), 30, 240, 208): (96, 96, 7, 2, 16),  # up3_res — swept 9364us
    # conv_out disabled — T_out_block=4 caused a frame-24-25 artifact; falls back to _DEFAULT_BLOCKINGS pending a clean re-sweep.
    # (2, 4, 96, 3, (3, 3, 3), 30, 240, 208): (96, 32, 4, 16, 2),  # conv_out — partial 5990us
    # ===================================================================
    # BH Galaxy 4x8, 720p image encoder, T=33 output frames
    # h_factor=4, w_factor=8. Per-device H/W are unpadded output dims.
    # ===================================================================
    # Stage 0 (full res, H_out=180, W_out=160)
    (4, 8, 32, 96, (3, 3, 3), 35, 180, 160): (32, 96, 5, 2, 16),  # conv_in — 3665us
    (4, 8, 96, 96, (3, 3, 3), 35, 180, 160): (96, 96, 3, 2, 16),  # res_s0  — 3907us
    (4, 8, 96, 96, (1, 3, 3), 33, 180, 160): (96, 96, 1, 2, 16),  # sp_s0   — 2787us
    # Stage 1 (half res, H_out=90, W_out=80)
    (4, 8, 96, 192, (3, 3, 3), 35, 90, 80): (96, 96, 7, 2, 16),  # down0   — 2403us
    (4, 8, 192, 192, (3, 3, 3), 35, 90, 80): (96, 96, 7, 4, 8),  # res_s1  — 4861us
    (4, 8, 192, 192, (1, 3, 3), 33, 90, 80): (192, 96, 1, 2, 16),  # sp_s1   — 1551us
    (4, 8, 192, 192, (3, 1, 1), 33, 45, 40): (192, 96, 7, 1, 32),  # tc_s1   — 318us PARTIAL
    # Stage 2 (quarter res, H_out=45, W_out=40)
    (4, 8, 192, 384, (3, 3, 3), 19, 45, 40): (96, 128, 3, 16, 2),  # down1   — TODO
    (4, 8, 384, 384, (3, 3, 3), 19, 45, 40): (96, 128, 5, 8, 4),  # res_s2  — TODO
    (4, 8, 384, 384, (1, 3, 3), 17, 45, 40): (192, 128, 1, 16, 2),  # sp_s2   — TODO
    (4, 8, 384, 384, (3, 1, 1), 17, 22, 20): (192, 384, 3, 8, 4),  # tc_s2   — TODO
    # Stage 3 (eighth res, H_out=22, W_out=20)
    (4, 8, 384, 384, (3, 3, 3), 11, 22, 20): (128, 96, 3, 8, 4),  # res_s3  — TODO
    (4, 8, 384, 32, (3, 3, 3), 11, 22, 20): (192, 32, 3, 8, 4),  # conv_out — TODO
    # ===================================================================
    # BH Galaxy 4x32, 720p image encoder, T=33 output frames
    # h_factor=4, w_factor=32. Per-device H/W are unpadded output dims.
    # Per-device (H,W): stage0(180,40) stage1(90,20) stage2(45,10) stage3(22,5)
    # Swept 2026-04-27 on 1x1 mesh; results in sweep_results_h4w32_enc_t33/.
    # hw_product=32 + max_t_block=8. T=5-7 wins. (16,2) dominant spatial.
    # ===================================================================
    # Stage 0 (full res, H_out=180, W_out=40)
    (4, 32, 32, 96, (3, 3, 3), 35, 180, 40): (32, 96, 6, 2, 16),  # conv_in_enc — 1089us
    (4, 32, 96, 96, (3, 3, 3), 35, 180, 40): (96, 96, 5, 8, 4),  # res_s0 — 1174us
    (4, 32, 96, 96, (1, 3, 3), 33, 180, 40): (96, 96, 1, 16, 2),  # sp_s0 — 970us
    # Stage 1 (half res, H_out=90, W_out=20)
    (4, 32, 96, 192, (3, 3, 3), 35, 90, 20): (96, 96, 5, 16, 2),  # down0 — 644us
    (4, 32, 192, 192, (3, 3, 3), 35, 90, 20): (96, 96, 6, 8, 4),  # res_s1 — 1235us
    (4, 32, 192, 192, (1, 3, 3), 33, 90, 20): (192, 96, 1, 8, 4),  # sp_s1 — 505us
    (4, 32, 192, 192, (3, 1, 1), 33, 45, 10): (192, 96, 5, 8, 4),  # tc_s1 — 130us
    # Stage 2 (quarter res, H_out=45, W_out=10)
    (4, 32, 192, 384, (3, 3, 3), 19, 45, 10): (96, 128, 3, 16, 2),  # down1 — 369us
    (4, 32, 384, 384, (3, 3, 3), 19, 45, 10): (128, 96, 3, 16, 2),  # res_s2 — 752us
    (4, 32, 384, 384, (1, 3, 3), 17, 45, 10): (384, 96, 1, 16, 2),  # sp_s2 — 217us
    (4, 32, 384, 384, (3, 1, 1), 17, 22, 5): (192, 384, 1, 8, 4),  # tc_s2 — 45us
    # Stage 3 (eighth res, H_out=22, W_out=5)
    (4, 32, 384, 384, (3, 3, 3), 11, 22, 5): (128, 64, 5, 8, 4),  # res_s3 — 200us
    (4, 32, 384, 32, (3, 3, 3), 11, 22, 5): (128, 32, 1, 8, 4),  # conv_out_enc — 52us
    # ===================================================================
    # BH Galaxy 4x8, 720p image encoder, T=16 output frames
    # Same blockings as T=33; T dims recomputed for encoder_t_chunk_size=16.
    # Stage 0: cur_T=16 → T_res=18, T_sp=16
    # Stage 1: cur_T=16 → T_res=18, T_tc=16, after_down=8
    # Stage 2: cur_T=8  → T_res=10, T_tc=8,  after_down=4
    # Stage 3: cur_T=4  → T_res=6
    # ===================================================================
    # Stage 0 (full res, H_out=180, W_out=160)
    (4, 8, 32, 96, (3, 3, 3), 18, 180, 160): (32, 96, 8, 2, 16),  # conv_in
    (4, 8, 96, 96, (3, 3, 3), 18, 180, 160): (96, 96, 7, 16, 2),  # res_s0
    (4, 8, 96, 96, (1, 3, 3), 16, 180, 160): (96, 96, 1, 2, 16),  # sp_s0
    # Stage 1 (half res, H_out=90, W_out=80)
    (4, 8, 96, 192, (3, 3, 3), 18, 90, 80): (96, 96, 8, 8, 4),  # down0
    (4, 8, 192, 192, (3, 3, 3), 18, 90, 80): (96, 96, 8, 16, 2),  # res_s1
    (4, 8, 192, 192, (1, 3, 3), 16, 90, 80): (192, 96, 1, 16, 2),  # sp_s1
    (4, 8, 192, 192, (3, 1, 1), 16, 45, 40): (192, 96, 3, 1, 32),  # tc_s1
    # Stage 2 (quarter res, H_out=45, W_out=40)
    (4, 8, 192, 384, (3, 3, 3), 10, 45, 40): (96, 128, 4, 16, 2),  # down1
    (4, 8, 384, 384, (3, 3, 3), 10, 45, 40): (128, 96, 3, 8, 4),  # res_s2
    (4, 8, 384, 384, (1, 3, 3), 8, 45, 40): (128, 384, 1, 4, 8),  # sp_s2
    (4, 8, 384, 384, (3, 1, 1), 8, 22, 20): (192, 384, 2, 8, 4),  # tc_s2
    # Stage 3 (eighth res, H_out=22, W_out=20)
    (4, 8, 384, 384, (3, 3, 3), 6, 22, 20): (128, 64, 4, 8, 4),  # res_s3
    (4, 8, 384, 32, (3, 3, 3), 6, 22, 20): (192, 32, 2, 8, 4),  # conv_out
    # ===================================================================
    # BH Galaxy 4x32, 720p image encoder, T=16 output frames
    # Same blockings as T=33; T dims recomputed for encoder_t_chunk_size=16.
    # Stage 0: cur_T=16 → T_res=18, T_sp=16
    # Stage 1: cur_T=16 → T_res=18, T_tc=16, after_down=8
    # Stage 2: cur_T=8  → T_res=10, T_tc=8,  after_down=4
    # Stage 3: cur_T=4  → T_res=6
    # ===================================================================
    # Stage 0 (full res, H_out=180, W_out=40)
    (4, 32, 32, 96, (3, 3, 3), 18, 180, 40): (32, 96, 8, 16, 2),  # conv_in_enc
    (4, 32, 96, 96, (3, 3, 3), 18, 180, 40): (96, 96, 6, 16, 2),  # res_s0
    (4, 32, 96, 96, (1, 3, 3), 16, 180, 40): (96, 96, 1, 2, 16),  # sp_s0
    # Stage 1 (half res, H_out=90, W_out=20)
    (4, 32, 96, 192, (3, 3, 3), 18, 90, 20): (96, 96, 4, 16, 2),  # down0
    (4, 32, 192, 192, (3, 3, 3), 18, 90, 20): (96, 96, 8, 8, 4),  # res_s1
    (4, 32, 192, 192, (1, 3, 3), 16, 90, 20): (192, 96, 1, 16, 2),  # sp_s1
    (4, 32, 192, 192, (3, 1, 1), 16, 45, 10): (192, 96, 5, 16, 2),  # tc_s1
    # Stage 2 (quarter res, H_out=45, W_out=10)
    (4, 32, 192, 384, (3, 3, 3), 10, 45, 10): (96, 96, 4, 8, 4),  # down1
    (4, 32, 384, 384, (3, 3, 3), 10, 45, 10): (128, 96, 3, 16, 2),  # res_s2
    (4, 32, 384, 384, (1, 3, 3), 8, 45, 10): (128, 384, 1, 16, 2),  # sp_s2
    (4, 32, 384, 384, (3, 1, 1), 8, 22, 5): (384, 96, 2, 8, 4),  # tc_s2 — T_out=4, capped from 5
    # Stage 3 (eighth res, H_out=22, W_out=5)
    (4, 32, 384, 384, (3, 3, 3), 6, 22, 5): (128, 64, 4, 16, 2),  # res_s3
    (4, 32, 384, 32, (3, 3, 3), 6, 22, 5): (192, 32, 1, 8, 4),  # conv_out_enc
    # ===================================================================
    # LTX-2.3 22B Video VAE decoder, BH Loud Box 2x4 (h_factor=2, w_factor=4), 1080p.
    # Regenerate via bruteforce_conv3d_sweep.py -k "sweep_all and h2w4"
    # ===================================================================
    (2, 4, 128, 1024, (3, 3, 3), 21, 17, 15): (64, 256, 1, 2, 16),  # ltx_s0_conv_in — 778us
    (2, 4, 1024, 1024, (3, 3, 3), 21, 17, 15): (128, 64, 5, 2, 16),  # ltx_s0_res — 7956us
    (2, 4, 1024, 4096, (3, 3, 3), 21, 17, 15): (128, 64, 5, 4, 8),  # ltx_s0_up — 22149us
    (2, 4, 512, 512, (3, 3, 3), 39, 34, 30): (64, 256, 1, 4, 8),  # ltx_s1_res — 8966us
    (2, 4, 512, 4096, (3, 3, 3), 39, 34, 30): (128, 64, 5, 2, 16),  # ltx_s1_up — 89486us
    (2, 4, 512, 512, (3, 3, 3), 75, 68, 60): (64, 256, 1, 8, 4),  # ltx_s2_res — 60810us
    (2, 4, 256, 256, (3, 3, 3), 147, 68, 60): (64, 256, 1, 8, 4),  # ltx_s3_res — 25688us
    (2, 4, 256, 512, (3, 3, 3), 147, 68, 60): (64, 256, 1, 8, 4),  # ltx_s3_chg — 48772us
    (2, 4, 128, 128, (3, 3, 3), 147, 136, 120): (64, 128, 6, 4, 8),  # ltx_s4_res — 22798us
    (2, 4, 128, 48, (3, 3, 3), 147, 136, 120): (128, 64, 6, 4, 8),  # ltx_s4_out — 13833us
    # LTX-2.3 spatial latent upsampler (x2), 2x4 BH-LB, 1080p.
    (2, 4, 128, 1024, (3, 3, 3), 21, 9, 8): (64, 256, 1, 2, 8),  # initial_conv
    (2, 4, 1024, 1024, (3, 3, 3), 21, 9, 8): (64, 32, 1, 2, 2),  # pre-upsample res
    (2, 4, 1024, 4096, (1, 3, 3), 19, 9, 8): (64, 32, 1, 2, 2),  # ups (kT=1)
    (2, 4, 1024, 1024, (3, 3, 3), 21, 18, 16): (64, 32, 1, 2, 2),  # post-upsample res
    (2, 4, 1024, 128, (3, 3, 3), 21, 18, 16): (64, 32, 1, 2, 2),  # final_conv
    # ===================================================================
    # LTX-2.3 22B Video VAE decoder, BH Galaxy 4x8 (h_factor=4, w_factor=8), 1080p.
    # Regenerate via bruteforce_conv3d_sweep.py -k "sweep_all and h4w8"
    # ===================================================================
    (4, 8, 128, 1024, (3, 3, 3), 21, 9, 8): (64, 128, 7, 8, 4),  # ltx_s0_conv_in — 237us
    (4, 8, 1024, 1024, (3, 3, 3), 21, 9, 8): (128, 64, 5, 4, 8),  # ltx_s0_res — 1974us
    (4, 8, 1024, 4096, (3, 3, 3), 21, 9, 8): (128, 64, 5, 4, 8),  # ltx_s0_up — 5448us
    (4, 8, 512, 512, (3, 3, 3), 39, 17, 15): (64, 256, 1, 4, 8),  # ltx_s1_res — 1912us
    (4, 8, 512, 4096, (3, 3, 3), 39, 17, 15): (128, 64, 5, 4, 8),  # ltx_s1_up — 16547us
    (4, 8, 512, 512, (3, 3, 3), 75, 34, 30): (64, 256, 1, 8, 4),  # ltx_s2_res — 13752us
    (4, 8, 256, 256, (3, 3, 3), 147, 34, 30): (64, 256, 1, 8, 4),  # ltx_s3_res — 6145us
    (4, 8, 256, 512, (3, 3, 3), 147, 34, 30): (64, 256, 1, 8, 4),  # ltx_s3_chg — 12013us
    (4, 8, 128, 128, (3, 3, 3), 147, 68, 60): (128, 64, 6, 2, 16),  # ltx_s4_res — 5647us
    (4, 8, 128, 48, (3, 3, 3), 147, 68, 60): (128, 64, 6, 2, 16),  # ltx_s4_out — 2914us
    # LTX-2.3 spatial latent upsampler (x2), BH Galaxy 4x8, 1080p.
    # Regenerate via bruteforce_conv3d_sweep.py -k "sweep_all and h4w8"
    (4, 8, 128, 1024, (3, 3, 3), 21, 5, 4): (128, 128, 3, 2, 4),  # ups_initial — 95us
    (4, 8, 1024, 1024, (3, 3, 3), 21, 5, 4): (128, 64, 7, 2, 4),  # ups_pre_res — 791us
    (4, 8, 1024, 4096, (1, 3, 3), 19, 5, 4): (256, 64, 1, 4, 4),  # ups_ups (kT=1) — 1235us
    (4, 8, 1024, 1024, (3, 3, 3), 21, 10, 8): (128, 64, 5, 4, 8),  # ups_post_res — 2012us
    (4, 8, 1024, 128, (3, 3, 3), 21, 10, 8): (128, 64, 7, 8, 4),  # ups_final — 277us
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
    # LTX-2.3 22B VAE decoder + latent upsampler conservative fallbacks. These
    # channel combos all have swept exact _BLOCKINGS entries for 2x4/4x8 1080p;
    # they remain here as the cross-mesh/cross-resolution fallback (the hardcoded
    # full-Cin default OOMs at these widths).
    (1024, 4096, (3, 3, 3)): (256, 32, 1, 1, 1),  # s0_up
    (512, 4096, (3, 3, 3)): (256, 32, 1, 1, 1),  # s1_up
    (256, 512, (3, 3, 3)): (256, 32, 1, 4, 4),  # s3_chg
    (128, 128, (3, 3, 3)): (128, 32, 1, 8, 8),  # s4_res
    (128, 48, (3, 3, 3)): (128, 32, 1, 8, 8),  # s4_out
    (1024, 4096, (1, 3, 3)): (256, 32, 1, 1, 1),  # upsampler (kT=1)
    (1024, 128, (3, 3, 3)): (256, 32, 1, 1, 1),  # upsampler final_conv
    # LTX-2.3 22B VAE encoder (I2V) conservative channel fallbacks.
    (64, 128, (3, 3, 3)): (64, 128, 1, 2, 2),  # conv_in
    (128, 64, (3, 3, 3)): (128, 64, 1, 2, 2),  # compress_space_res inner conv (mult 2)
    (256, 256, (3, 3, 3)): (128, 32, 1, 2, 2),  # res_x @256 / compress_time_res inner conv
    (512, 128, (3, 3, 3)): (256, 32, 1, 1, 1),  # compress_all_res inner conv (mult 2)
    (512, 512, (3, 3, 3)): (256, 32, 1, 1, 1),  # res_x @512
    (1024, 1024, (3, 3, 3)): (256, 32, 1, 1, 1),  # res_x @1024 / compress_all_res inner conv (mult 1)
    # LTX-2.3 audio mel-VAE decoder (Conv2dViaConv3d, kT=1, W=mel_bins=16). These
    # combos otherwise fell to the hardcoded (Cin, 32, 1, 1, 1) default, whose
    # H_out=W_out=1 forces one output pixel per work-unit (4-16ms/conv). Block the
    # full mel width (W_out=16) and a height chunk so each unit produces more output.
    (512, 512, (1, 3, 3)): (512, 32, 1, 8, 16),
    (256, 256, (1, 3, 3)): (256, 32, 1, 8, 16),
    (128, 128, (1, 3, 3)): (128, 32, 1, 8, 16),
    (512, 256, (1, 3, 3)): (512, 32, 1, 8, 16),
    (256, 128, (1, 3, 3)): (256, 32, 1, 8, 16),
    (32, 512, (1, 3, 3)): (32, 32, 1, 8, 16),  # conv_in (z_channels 8 -> aligned 32)
    (128, 32, (1, 3, 3)): (128, 32, 1, 8, 16),  # conv_out (out_ch 2 -> aligned 32)
}


# fp32 conv3d blockings (separate table; 2× L1 vs bf16).
# Key: (in_channels, out_channels, kernel_size). Value: (C_in, C_out, T_out, 1, 1).
_FP32_BLOCKINGS: dict = {
    # ups inner-conv
    (1536, 768, (11, 1, 1)): (128, 128, 32, 1, 1),  # ups[0]
    (768, 384, (4, 1, 1)): (256, 32, 64, 1, 1),  # ups[1]
    (384, 192, (4, 1, 1)): (128, 64, 32, 1, 1),  # ups[2]
    (192, 96, (4, 1, 1)): (64, 32, 64, 1, 1),  # ups[3]
    (96, 64, (4, 1, 1)): (32, 32, 16, 1, 1),  # ups[4]
    (64, 32, (4, 1, 1)): (64, 32, 16, 1, 1),  # ups[5]
    # main vocoder
    (128, 1536, (7, 1, 1)): (128, 32, 29, 1, 1),  # conv_pre
    (768, 768, (11, 1, 1)): (128, 64, 64, 1, 1),  # stage 0 AMP k11
    (768, 768, (7, 1, 1)): (256, 32, 64, 1, 1),  # stage 0 AMP k7
    (768, 768, (3, 1, 1)): (128, 64, 16, 1, 1),  # stage 0 AMP k3
    (384, 384, (11, 1, 1)): (32, 64, 32, 1, 1),  # stage 1 AMP k11
    (384, 384, (7, 1, 1)): (128, 128, 16, 1, 1),  # stage 1 AMP k7
    (384, 384, (3, 1, 1)): (64, 384, 8, 1, 1),  # stage 1 AMP k3
    (192, 192, (11, 1, 1)): (32, 32, 16, 1, 1),  # stage 2 AMP k11
    (192, 192, (7, 1, 1)): (64, 64, 16, 1, 1),  # stage 2 AMP k7
    (192, 192, (3, 1, 1)): (64, 32, 16, 1, 1),  # stage 2 AMP k3
    (96, 96, (11, 1, 1)): (32, 32, 16, 1, 1),  # stage 3 AMP k11
    (96, 96, (7, 1, 1)): (32, 32, 16, 1, 1),  # stage 3 AMP k7
    (96, 96, (3, 1, 1)): (32, 32, 8, 1, 1),  # stage 3 AMP k3
    (64, 64, (11, 1, 1)): (64, 64, 32, 1, 1),  # stage 4 AMP k11
    (64, 64, (7, 1, 1)): (64, 64, 4, 1, 1),  # stage 4 AMP k7
    (64, 64, (3, 1, 1)): (32, 32, 16, 1, 1),  # stage 4 AMP k3
    (32, 32, (11, 1, 1)): (32, 32, 4, 1, 1),  # stage 5 AMP k11
    (32, 32, (7, 1, 1)): (32, 32, 64, 1, 1),  # stage 5 AMP k7
    (32, 32, (3, 1, 1)): (32, 32, 2, 1, 1),  # stage 5 AMP k3
    # BWE vocoder
    (128, 512, (7, 1, 1)): (64, 32, 7, 1, 1),  # conv_pre
    # BWE upsample inner-convs. Without these they hit the default (…,1,1,1): T_out_block=1
    # collapses the matmul M dim to a single 1/32-full tile and reloads the weight once per
    # output frame, so the long-T ups cost ~130/66/15 ms. T_out_block=32 fills the M tile.
    (512, 256, (12, 1, 1)): (128, 128, 32, 1, 1),  # ups[0] rate6
    (256, 128, (11, 1, 1)): (128, 64, 32, 1, 1),  # ups[1] rate5
    (128, 64, (4, 1, 1)): (64, 32, 32, 1, 1),  # ups[2] rate2
    (64, 32, (4, 1, 1)): (64, 32, 32, 1, 1),  # ups[3] rate2
    (32, 32, (4, 1, 1)): (32, 32, 32, 1, 1),  # ups[4] rate2 (finest T)
    (256, 256, (11, 1, 1)): (64, 32, 6, 1, 1),  # stage 0 AMP k11
    (256, 256, (7, 1, 1)): (256, 32, 3, 1, 1),  # stage 0 AMP k7
    (256, 256, (3, 1, 1)): (256, 128, 4, 1, 1),  # stage 0 AMP k3
    (128, 128, (11, 1, 1)): (128, 64, 3, 1, 1),  # stage 1 AMP k11
    (128, 128, (7, 1, 1)): (128, 128, 2, 1, 1),  # stage 1 AMP k7
    (128, 128, (3, 1, 1)): (128, 128, 4, 1, 1),  # stage 1 AMP k3
    (64, 64, (11, 1, 1)): (32, 64, 15, 1, 1),  # stage 2 AMP k11
    (64, 64, (7, 1, 1)): (64, 64, 3, 1, 1),  # stage 2 AMP k7
    (64, 64, (3, 1, 1)): (64, 64, 3, 1, 1),  # stage 2 AMP k3
    (32, 32, (11, 1, 1)): (32, 32, 3, 1, 1),  # stage 3 AMP k11
    (32, 32, (7, 1, 1)): (32, 32, 4, 1, 1),  # stage 3 AMP k7
    (32, 32, (3, 1, 1)): (32, 32, 5, 1, 1),  # stage 3 AMP k3
    # 16 channels: aligned(16)=32 → key (32,32,...) covered by stage 3 above.
}


def register_conv3d_configs(configs: dict) -> None:
    """Register additional conv3d blocking configs from external models.

    Entries are added to the fallback table keyed by ``(in_channels, out_channels, kernel_size)``.

    Args:
        configs: Mapping from ``(in_channels, out_channels, kernel_size)``
            to ``(C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)``.

    Example::

        register_conv3d_configs({
            (32, 96, (3, 3, 3)): (32, 96, 1, 8, 16),
            (384, 768, (3, 1, 1)): (384, 384, 1, 16, 4),
        })
    """
    _DEFAULT_BLOCKINGS.update({(c_in, c_out, _ntuple(ks, 3)): tuple(v) for (c_in, c_out, ks), v in configs.items()})


def get_conv3d_config(
    in_channels, out_channels, kernel_size, weights_dtype, grid_size, *, h_factor=1, w_factor=1, T=0, H=0, W=0
):
    """Get optimized Conv3dConfig for a conv3d layer.

    Lookup chain: exact (mesh, shape, T, spatial) match -> fallback (channel, kernel) match -> default.
    Pass h_factor, w_factor, T, H, W for best results. When these are not
    available the fallback table is used.
    """
    if weights_dtype == ttnn.float32:
        fp32_blk = _FP32_BLOCKINGS.get((in_channels, out_channels, kernel_size))
        if fp32_blk is not None:
            C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = fp32_blk
            logger.debug(f"conv3d fp32 blocking [exact] ({in_channels},{out_channels},{kernel_size}) -> {fp32_blk}")
        else:
            # Conservative default — unchanged from the prior hardcoded fp32 path.
            C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = 32, 32, 1, 1, 1
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

    blocking_key = (h_factor, w_factor, in_channels, out_channels, kernel_size, T, H, W)
    channel_key = (in_channels, out_channels, kernel_size)

    exact = _BLOCKINGS.get(blocking_key)
    if exact is not None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = exact
        logger.debug(
            f"conv3d blocking [exact] {blocking_key} -> "
            f"Cin={C_in_block} Cout={C_out_block} T={T_out_block} H={H_out_block} W={W_out_block}"
        )
    else:
        fallback = _DEFAULT_BLOCKINGS.get(channel_key)
        if fallback is not None:
            C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = fallback
            logger.warning(
                f"conv3d blocking [fallback] {blocking_key} -> channel_key={channel_key} -> "
                f"Cin={C_in_block} Cout={C_out_block} T={T_out_block} H={H_out_block} W={W_out_block}"
            )
        else:
            C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = in_channels, 32, 1, 1, 1
            logger.warning(
                f"conv3d blocking [NONE] {blocking_key} -> no match in any table, using hardcoded default: "
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
    """Build a cache key suffix from the per-conv weight-prep state of all conv3d layers.

    Cached weights depend on C_in_block (prepare_conv3d_weights reshapes by it) and, for
    depth-to-space convs, depth_to_space_stride (they reorder output channels at load). The
    stride is appended only when set, so keys for modules without any depth-to-space conv are
    byte-identical to the pre-reorder scheme and their existing caches stay valid.
    """
    parts = []
    for m in _walk_conv3d_modules(module):
        dts = getattr(m, "depth_to_space_stride", None)
        parts.append(str(m.conv_config.C_in_block) if dts is None else f"{m.conv_config.C_in_block}:{dts}")
    if not parts:
        return ""
    return "cin" + hashlib.sha256("_".join(parts).encode()).hexdigest()[:8]


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


def conv_pad_width(tensor_BTHWC, w_factor):
    """
    Pad the width to the next multiple of w_factor, mirroring conv_pad_height.
    """
    B, T, H, W, C = tensor_BTHWC.shape
    pad_w = (w_factor - W % w_factor) % w_factor
    if pad_w > 0:
        tensor_BTHWC = torch.nn.functional.pad(tensor_BTHWC, (0, 0, 0, pad_w))
    return tensor_BTHWC, W


def conv_unpad_width(tensor_BTHWC, logical_w):
    """
    Remove width padding that was added by conv_pad_width.
    """
    return tensor_BTHWC[:, :, :, :logical_w, :]


def conv_pad_in_channels(tensor):
    C_in = tensor.shape[-1]
    padded_C_in = aligned_channels(C_in)
    if padded_C_in != C_in:
        tensor = torch.nn.functional.pad(tensor, (0, padded_C_in - C_in))
    return tensor
