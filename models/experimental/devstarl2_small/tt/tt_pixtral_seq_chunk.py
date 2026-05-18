# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Shared Pixtral vision sequence chunk size for L1-bound matmuls (attention + MLP).

from __future__ import annotations

import os

from models.common.utility_functions import nearest_32


def pixtral_vision_seq_chunk_len(configuration) -> int:
    """Tokens per chunk along the vision sequence axis for TT ops whose matmul programs grow L1 CB with ``m``. - **PIXTRAL_VISION_MM_SEQ_CHUNK**: if set, use only this (tile-rounded). Ignores ``VISION_MAX_MM_SEQ``. - Else: ``min(VISION_MAX_MM_SEQ, PIXTRAL_VISION_MM_SEQ_CHUNK_CAP)`` (default cap 448). **PIXTRAL_VISION_SEQ_CHUNK_DEBUG=1**: print resolved chunk (once per distinct value)."""
    force = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK")
    if force is not None and str(force).strip() != "":
        chunk = max(32, nearest_32(int(force)))
    else:
        cap_raw = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK_CAP", "448")
        cap = max(32, nearest_32(int(cap_raw)))
        cfg_chunk = getattr(configuration, "VISION_MAX_MM_SEQ", cap)
        if cfg_chunk is None:
            cfg_chunk = cap
        chunk = max(32, min(int(cfg_chunk), int(cap)))

    if os.environ.get("PIXTRAL_VISION_SEQ_CHUNK_DEBUG", "").strip() in ("1", "true", "yes"):
        if getattr(pixtral_vision_seq_chunk_len, "_debug_chunk_announced", None) != chunk:
            print(
                f"[pixtral] vision seq_chunk_len={chunk} "
                f"(PIXTRAL_VISION_MM_SEQ_CHUNK={os.environ.get('PIXTRAL_VISION_MM_SEQ_CHUNK')!r}, "
                f"CAP={os.environ.get('PIXTRAL_VISION_MM_SEQ_CHUNK_CAP')!r})",
                flush=True,
            )
            pixtral_vision_seq_chunk_len._debug_chunk_announced = chunk

    return chunk
