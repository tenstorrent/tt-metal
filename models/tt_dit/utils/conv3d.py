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
    target_height, target_width, h_factor, w_factor, t_chunk_size, *, temperal_upsample, num_stages=3, cached=False
):
    """Compute per-stage spatial and temporal dimensions for a VAE decoder.

    Returns (stage_hw, stage_t) where:
      stage_hw: list of StageHW per stage (length num_stages + 1), latent -> full resolution
      stage_t:  list of StageT per stage (length num_stages + 1)

    Temporal formulas differ between uncached and cached paths because
    WanResample splits off frame-0 in the uncached path but not in the cached path:

      uncached: T_tconv = cur_T + 1   (frames[1:] = cur_T-1, +2 causal pad)
                T_spatial = 2*(cur_T-1) + 1   (frame-0 + doubled rest)
      cached:   T_tconv = cur_T + 2   (all frames + 2 causal pad from cache)
                T_spatial = 2 * cur_T         (all frames doubled)
    """
    if t_chunk_size is None or t_chunk_size < 1:
        n = num_stages + 1
        return [StageHW(0, 0)] * n, [StageT(T_res=0, T_tconv=0, T_spatial=0)] * n

    vae_scale = 2**num_stages
    # Height uses ceil because some configs don't divide evenly (e.g. 720/8/4=22.5 → 23);
    # the hardware pads height via conv_pad_height.  Width always divides evenly for
    # production targets and is not padded the same way, so floor division is correct.
    lat_h = math.ceil(target_height / vae_scale / h_factor)
    lat_w = target_width // vae_scale // w_factor
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
        stage_t.append(
            StageT(
                T_res=cur_T + 2,
                T_tconv=T_tconv,
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
    # BH Galaxy 6U 4x32, 480p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(15,3) stage1(30,6) stage2(60,12) stage3(120,24)
    # Padded (int_pad=(0,1,1)): stage0(17,5) stage1(32,8) stage2(62,14) stage3(122,26)
    # Swept 2026-04-13 on 1x1 mesh; results in sweep_results_h4w32_480p_full_t/.
    # hw_product=32 + max_t_block=8. T=3-7 wins. Large-C_in tconv/res partial.
    # ===================================================================
    (4, 32, 32, 384, (3, 3, 3), 23, 15, 3): (32, 64, 3, 16, 2),  # conv_in — 141us
    (4, 32, 384, 384, (3, 3, 3), 23, 15, 3): (96, 64, 6, 8, 4),  # lat_mid_res — 377us
    (4, 32, 384, 768, (3, 1, 1), 22, 15, 3): (192, 256, 1, 8, 4),  # up0_tconv — 240us table
    (4, 32, 384, 192, (1, 3, 3), 41, 30, 6): (96, 96, 1, 16, 2),  # up0_spatial — 354us
    (4, 32, 192, 384, (3, 3, 3), 43, 30, 6): (64, 128, 6, 4, 8),  # up1_res0 — 725us
    (4, 32, 384, 384, (3, 3, 3), 43, 30, 6): (128, 96, 3, 16, 2),  # up1_res — 1313us partial
    (4, 32, 384, 768, (3, 1, 1), 42, 30, 6): (192, 256, 5, 16, 2),  # up1_tconv — 366us partial
    (4, 32, 384, 192, (1, 3, 3), 81, 60, 12): (384, 64, 1, 16, 2),  # up1_spatial — 1237us
    (4, 32, 192, 192, (3, 3, 3), 83, 60, 12): (96, 96, 3, 16, 2),  # up2_res — 1851us
    (4, 32, 192, 96, (1, 3, 3), 81, 120, 24): (192, 96, 1, 8, 4),  # up2_spatial — 1398us table
    (4, 32, 96, 96, (3, 3, 3), 83, 120, 24): (96, 96, 7, 4, 8),  # up3_res — 1684us
    (4, 32, 96, 3, (3, 3, 3), 83, 120, 24): (96, 32, 3, 16, 2),  # conv_out — 1421us
    # ===================================================================
    # BH Galaxy 6U 4x32, 720p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(23,5) stage1(46,10) stage2(92,20) stage3(184,40)
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
    # BH Galaxy 4x8, 480p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(15,13) stage1(30,26) stage2(60,52) stage3(120,104)
    # Padded (int_pad=(0,1,1)): stage0(17,15) stage1(32,28) stage2(62,54) stage3(122,106)
    # Swept 2026-04-10 on 1x1 mesh; results in sweep_results_h4w8_480p_full_t/.
    # hw_product=32 + max_t_block=8. Partial: large-C_in/tconv layers hung mid-sweep.
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 23, 15, 13): (32, 128, 3, 4, 8),  # conv_in — 197us
    (4, 8, 384, 384, (3, 3, 3), 23, 15, 13): (128, 64, 3, 16, 2),  # lat_mid_res — 711us
    (4, 8, 384, 768, (3, 1, 1), 22, 15, 13): (192, 128, 5, 4, 8),  # up0_tconv — 277us partial
    (4, 8, 384, 192, (1, 3, 3), 41, 30, 26): (192, 64, 1, 16, 2),  # up0_spatial — 790us
    (4, 8, 192, 384, (3, 3, 3), 43, 30, 26): (64, 128, 6, 16, 2),  # up1_res0 — 1911us partial
    (4, 8, 384, 384, (3, 3, 3), 43, 30, 26): (96, 96, 3, 16, 2),  # up1_res — 4069us partial
    (4, 8, 384, 768, (3, 1, 1), 42, 30, 26): (384, 384, 1, 16, 2),  # up1_tconv — 1110us partial
    (4, 8, 384, 192, (1, 3, 3), 81, 60, 52): (384, 64, 1, 8, 4),  # up1_spatial — 5030us
    (4, 8, 192, 192, (3, 3, 3), 83, 60, 52): (96, 96, 3, 16, 2),  # up2_res — 7547us
    (4, 8, 192, 96, (1, 3, 3), 81, 120, 104): (192, 96, 1, 8, 4),  # up2_spatial — table 5755us
    (4, 8, 96, 96, (3, 3, 3), 83, 120, 104): (96, 96, 7, 4, 8),  # up3_res — 6839us
    (4, 8, 96, 3, (3, 3, 3), 83, 120, 104): (96, 32, 8, 4, 4),  # conv_out — 4986us
    # ===================================================================
    # BH Galaxy 4x8, 720p, 81 frames full-T (latent T=21)
    # Per-device (H,W): stage0(23,20) stage1(46,40) stage2(92,80) stage3(184,160)
    # Swept 2026-04-13 on 1x1 mesh; results in sweep_results_h4w8_720p_full_t/.
    # hw_product=32 + max_t_block=8. T=3-6 wins. tconv layers partial.
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 23, 23, 20): (32, 128, 6, 8, 4),  # conv_in — 294us
    (4, 8, 384, 384, (3, 3, 3), 23, 23, 20): (96, 96, 1, 8, 4),  # lat_mid_res — 1289us table
    (4, 8, 384, 768, (3, 1, 1), 22, 23, 20): (192, 384, 1, 8, 4),  # up0_tconv — 484us partial
    (4, 8, 384, 192, (1, 3, 3), 41, 46, 40): (384, 64, 1, 16, 2),  # up0_spatial — 1516us
    (4, 8, 192, 384, (3, 3, 3), 43, 46, 40): (96, 128, 1, 16, 2),  # up1_res0 — 4163us
    (4, 8, 384, 384, (3, 3, 3), 43, 46, 40): (96, 96, 6, 16, 2),  # up1_res — 9438us
    (4, 8, 384, 768, (3, 1, 1), 42, 46, 40): (384, 384, 1, 4, 8),  # up1_tconv — 2316us partial
    (4, 8, 384, 192, (1, 3, 3), 81, 92, 80): (384, 64, 1, 4, 8),  # up1_spatial — 11917us
    (4, 8, 192, 192, (3, 3, 3), 83, 92, 80): (96, 96, 3, 4, 8),  # up2_res — 17238us
    (4, 8, 192, 96, (1, 3, 3), 81, 184, 160): (192, 96, 1, 8, 4),  # up2_spatial — 13648us table
    (4, 8, 96, 96, (3, 3, 3), 83, 184, 160): (96, 96, 4, 4, 16),  # up3_res — 16056us (hw=64)
    (4, 8, 96, 3, (3, 3, 3), 83, 184, 160): (96, 32, 3, 16, 2),  # conv_out — 11294us
    # ===================================================================
    # BH Galaxy 4x8, 720p, cached t_chunk_size=1 (vae_t_chunk_size=1)
    # First chunk / anchor frame: all stages see T = t_chunk + 2 = 3.
    # Swept 2026-04-11 on 1x1 mesh; results in sweep_results_h4w8_720p_t1/.
    # T_out=1 for (3,3,3) → only T_block=1 tested. Large-C_in tconv layers partial.
    # ===================================================================
    (4, 8, 32, 384, (3, 3, 3), 3, 23, 20): (32, 64, 1, 8, 4),  # conv_in — 111us
    (4, 8, 384, 384, (3, 3, 3), 3, 23, 20): (64, 96, 1, 8, 4),  # lat_mid_res — 253us
    (4, 8, 384, 768, (3, 1, 1), 3, 23, 20): (192, 96, 1, 4, 8),  # up0_tconv — 132us partial
    (4, 8, 384, 192, (1, 3, 3), 3, 46, 40): (128, 64, 1, 16, 2),  # up0_spatial — 261us
    (4, 8, 192, 384, (3, 3, 3), 3, 46, 40): (64, 96, 1, 16, 2),  # up1_res0 — 318us
    (4, 8, 384, 384, (3, 3, 3), 3, 46, 40): (64, 128, 1, 16, 2),  # up1_res — 474us
    (4, 8, 384, 768, (3, 1, 1), 3, 46, 40): (128, 256, 1, 4, 8),  # up1_tconv — 226us partial
    (4, 8, 384, 192, (1, 3, 3), 3, 92, 80): (128, 96, 1, 16, 2),  # up1_spatial — 583us
    (4, 8, 192, 192, (3, 3, 3), 3, 92, 80): (96, 96, 1, 16, 2),  # up2_res — 455us
    (4, 8, 192, 96, (1, 3, 3), 3, 184, 160): (192, 96, 1, 16, 2),  # up2_spatial — 615us
    (4, 8, 96, 96, (3, 3, 3), 3, 184, 160): (96, 96, 1, 8, 4),  # up3_res — table 492us
    (4, 8, 96, 3, (3, 3, 3), 3, 184, 160): (96, 32, 1, 8, 4),  # conv_out — 393us
    # ===================================================================
    # BH Galaxy 4x8, 720p, cached t_chunk_size=15 (vae_t_chunk_size=15)
    # Stage 0: T_res=17, T_tconv=17. Stage 1: T_res=32, T_tconv=32, T_sp=60.
    # Stage 2/3: T_res=62. Spatial bridge T: T_sp_stage0=30, T_sp_stage1=60.
    # Swept 2026-04-11 on 1x1 mesh; results in sweep_results_h4w8_720p_t15/.
    # hw_product=32 + max_t_block=8. T=3-7 win. Large-C_in tconv layers partial.
    # ===================================================================
    # Stage 0 (cur_T=15)
    (4, 8, 32, 384, (3, 3, 3), 17, 23, 20): (32, 128, 3, 16, 2),  # conv_in — 265us
    (4, 8, 384, 384, (3, 3, 3), 17, 23, 20): (96, 64, 3, 8, 4),  # lat_mid_res — 1018us partial
    (4, 8, 384, 768, (3, 1, 1), 17, 23, 20): (128, 384, 1, 8, 4),  # up0_tconv — 448us partial
    (4, 8, 384, 192, (1, 3, 3), 30, 46, 40): (192, 96, 1, 16, 2),  # up0_spatial — 1007us
    # Stage 1 (cur_T=30)
    (4, 8, 192, 384, (3, 3, 3), 32, 46, 40): (96, 128, 3, 4, 8),  # up1_res0 — 2968us
    (4, 8, 384, 384, (3, 3, 3), 32, 46, 40): (128, 96, 3, 16, 2),  # up1_res — 6086us
    (4, 8, 384, 768, (3, 1, 1), 32, 46, 40): (192, 768, 1, 8, 4),  # up1_tconv — 1603us partial
    (4, 8, 384, 192, (1, 3, 3), 60, 92, 80): (192, 96, 1, 4, 8),  # up1_spatial — 7948us
    # Stage 2/3 (cur_T=60, no temporal upsample)
    (4, 8, 192, 192, (3, 3, 3), 62, 92, 80): (96, 96, 6, 4, 8),  # up2_res — 11756us
    (4, 8, 192, 96, (1, 3, 3), 60, 184, 160): (192, 96, 1, 4, 8),  # up2_spatial — 8000us
    (4, 8, 96, 96, (3, 3, 3), 62, 184, 160): (96, 96, 7, 2, 16),  # up3_res — 12020us
    (4, 8, 96, 3, (3, 3, 3), 62, 184, 160): (96, 32, 4, 16, 2),  # conv_out — 8212us
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
    (4, 8, 32, 384, (3, 3, 3), 18, 23, 20): (32, 128, 4, 4, 8),  # conv_in — 253us
    (4, 8, 384, 384, (3, 3, 3), 18, 23, 20): (96, 96, 2, 8, 4),  # lat_mid_res — 991us partial
    (4, 8, 384, 768, (3, 1, 1), 18, 23, 20): (192, 384, 1, 8, 4),  # up0_tconv — 474us partial
    (4, 8, 384, 192, (1, 3, 3), 32, 46, 40): (192, 96, 1, 16, 2),  # up0_spatial — 1055us
    # Stage 1 (cur_T=32)
    (4, 8, 192, 384, (3, 3, 3), 34, 46, 40): (96, 96, 2, 8, 4),  # up1_res0 — 3359us partial
    (4, 8, 384, 384, (3, 3, 3), 34, 46, 40): (96, 96, 8, 8, 4),  # up1_res — 6502us partial
    (4, 8, 384, 768, (3, 1, 1), 34, 46, 40): (192, 768, 1, 4, 8),  # up1_tconv — 1824us partial
    (4, 8, 384, 192, (1, 3, 3), 64, 92, 80): (384, 96, 1, 4, 8),  # up1_spatial — 7914us
    # Stage 2 (cur_T=64, no temporal upsample)
    (4, 8, 192, 192, (3, 3, 3), 66, 92, 80): (96, 96, 4, 16, 2),  # up2_res — 12401us
    (4, 8, 192, 96, (1, 3, 3), 64, 184, 160): (192, 96, 1, 4, 8),  # up2_spatial — 7914us
    # Stage 3 (cur_T=64, no temporal upsample)
    (4, 8, 96, 96, (3, 3, 3), 66, 184, 160): (96, 96, 8, 4, 8),  # up3_res — 11980us
    (4, 8, 96, 3, (3, 3, 3), 66, 184, 160): (96, 32, 3, 16, 2),  # conv_out — 8887us
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
    (2, 4, 32, 384, (3, 3, 3), 9, 30, 26): (32, 128, 7, 2, 2),  # conv_in — swept 244us
    (2, 4, 384, 384, (3, 3, 3), 9, 30, 26): (96, 96, 1, 32, 4),  # lat_mid_res — partial 1009us
    (2, 4, 384, 768, (3, 1, 1), 9, 30, 26): (192, 256, 1, 16, 2),  # up0_tconv — partial 417us
    (2, 4, 384, 192, (1, 3, 3), 14, 60, 52): (192, 96, 1, 32, 4),  # up0_spatial — table wins 1034us
    # Stage 1 (cur_T=14): T_res=16, T_tconv=16, T_spatial=28
    (2, 4, 192, 384, (3, 3, 3), 16, 60, 52): (96, 96, 7, 16, 2),  # up1_res0 — partial 2446us
    (2, 4, 384, 384, (3, 3, 3), 16, 60, 52): (96, 96, 7, 16, 2),  # up1_res — inferred from up1_res0
    (2, 4, 384, 768, (3, 1, 1), 16, 60, 52): (192, 768, 1, 8, 4),  # up1_tconv — swept 1442us
    (2, 4, 384, 192, (1, 3, 3), 28, 120, 104): (384, 96, 1, 4, 8),  # up1_spatial — swept 6809us
    # Stage 2 (cur_T=28): T_res=30, T_spatial=28 (no temporal upsample)
    (2, 4, 192, 192, (3, 3, 3), 30, 120, 104): (96, 96, 7, 4, 8),  # up2_res — swept 9400us
    (2, 4, 192, 96, (1, 3, 3), 28, 240, 208): (192, 96, 1, 4, 16),  # up2_spatial — swept 6509us
    # Stage 3 (cur_T=28): T_res=30 (no temporal upsample)
    (2, 4, 96, 96, (3, 3, 3), 30, 240, 208): (96, 96, 7, 2, 16),  # up3_res — swept 9364us
    (2, 4, 96, 3, (3, 3, 3), 30, 240, 208): (96, 32, 4, 16, 2),  # conv_out — partial 5990us
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
