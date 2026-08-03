# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device capability helpers for XTTS on Blackhole.

P150 (single-chip) exposes compute grid ``(11, 10)`` and ~1.5 MB L1/core. Longer demos
overflow that L1 when activations stay on-chip (GPT prefill residuals, HiFi-GAN conv CBs).
Larger Blackhole cards / lab grids (e.g. gx>=12 used by the original matmul pins) have more
room — keep the fast L1 paths there. All L1 fallbacks are gated on :func:`is_p150_device`.
"""

from __future__ import annotations

import ttnn

# Measured P150 compute grid from TT_FATAL / Tracy on this SKU.
_P150_GRID = (11, 10)

# Prefill: bf16 [1, S, 1024] residuals + LN CBs overflow P150 L1 past this length.
_P150_PREFILL_L1_MAX_SEQ = 160

# Vocoder: interleaved conv_pre DB/auto act_block_h overflows P150 L1 past this upsampled length.
_P150_VOCODER_L1_SAFE_LENGTH = 256

# Resample matmul in0: keep small matrices in L1 on P150; larger ones go DRAM.
_P150_RESAMPLE_L1_BUDGET = 256 * 1024

# L1_SMALL for halo/config tensors. Demo opens at 64 KB; long P150 vocoder chains need more
# headroom even with config_tensors_in_dram (other ops still touch L1_SMALL).
_P150_L1_SMALL_SIZE = 131072
_DEFAULT_L1_SMALL_SIZE = 65536


def compute_grid_xy(device) -> tuple[int, int]:
    grid = device.compute_with_storage_grid_size()
    return int(grid.x), int(grid.y)


def is_p150_device(device) -> bool:
    """True for Blackhole P150 compute geometry ``(11, 10)``.

    Detected from the open device (not tt-smi / host board name), so it works in-process
    after ``open_device`` and stays correct on multi-chip hosts when each chip is a P150.
    Larger BH grids (gx>11) return False and keep the original L1-aggressive configs.
    """
    return compute_grid_xy(device) == _P150_GRID


def recommended_l1_small_size(device=None) -> int:
    """``open_device(..., l1_small_size=)`` for XTTS.

    P150 long vocodes need ~128 KB. When ``device`` is unknown (demo open path), use the
    P150-safe size — larger BH cards tolerate the extra reservation.
    """
    if device is None or is_p150_device(device):
        return _P150_L1_SMALL_SIZE
    return _DEFAULT_L1_SMALL_SIZE


def prefill_act_memory_config(device, seq_len: int):
    """Activation memory for GPT prefill: DRAM on long P150 prompts, else L1."""
    if is_p150_device(device) and int(seq_len) > _P150_PREFILL_L1_MAX_SEQ:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def needs_vocoder_l1_safe(device, input_length: int) -> bool:
    """Whether interleaved HiFi-GAN conv1d should drop DB / cap act_block_h (P150 only)."""
    return is_p150_device(device) and int(input_length) > _P150_VOCODER_L1_SAFE_LENGTH


def resample_matrix_memory_config(device, nbytes: int):
    """Latent-upsample resample matrix placement: tight L1 budget on P150 only."""
    budget = _P150_RESAMPLE_L1_BUDGET if is_p150_device(device) else (1024 * 1024)
    return ttnn.L1_MEMORY_CONFIG if int(nbytes) <= budget else ttnn.DRAM_MEMORY_CONFIG
