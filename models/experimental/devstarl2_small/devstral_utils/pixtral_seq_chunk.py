# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral vision sequence chunk size for L1-bound matmuls (attention + MLP).

from __future__ import annotations

import os

import ttnn

from models.common.utility_functions import nearest_32


def pixtral_vision_seq_chunk_len(configuration) -> int:
    """L1 matmul seq chunk (env PIXTRAL_VISION_MM_SEQ_CHUNK or min(cfg, cap))."""
    force = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK")
    if force is not None and str(force).strip() != "":
        chunk = max(32, nearest_32(int(force)))
    else:
        cap_raw = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK_CAP", "1024")
        cap = max(32, nearest_32(int(cap_raw)))
        cfg_chunk = getattr(configuration, "VISION_MAX_MM_SEQ", cap)
        if cfg_chunk is None:
            cfg_chunk = cap
        chunk = max(32, min(int(cfg_chunk), int(cap)))

    return chunk


def pixtral_effective_mm_seq_len(configuration, seq_len: int) -> int:
    """Matmul M: one kernel over full ``seq_len`` when it fits L1, else ``pixtral_vision_seq_chunk_len``."""
    chunk = pixtral_vision_seq_chunk_len(configuration)
    force = os.environ.get("PIXTRAL_VISION_MM_FULL_SEQ_CAP")
    if force is not None and str(force).strip() != "":
        full_cap = max(32, nearest_32(int(force)))
    else:
        full_cap = max(chunk, 1024)
    if seq_len <= full_cap:
        return seq_len
    return chunk


def pad_seq_to_chunk_multiple(x: ttnn.Tensor, seq_len: int, chunk: int) -> tuple[ttnn.Tensor, int, int]:
    """Pad dim=2 to a multiple of ``chunk`` so batched matmul avoids ``ttnn.concat``."""
    original = seq_len
    if seq_len <= chunk or seq_len % chunk == 0:
        return x, seq_len, original
    padded = ((seq_len + chunk - 1) // chunk) * chunk
    pad_len = padded - seq_len
    x = ttnn.pad(x, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
    return x, padded, original


def trim_seq_dim2(x: ttnn.Tensor, original_seq_len: int) -> ttnn.Tensor:
    if int(x.shape[2]) == original_seq_len:
        return x
    return x[:, :, :original_seq_len, :]
