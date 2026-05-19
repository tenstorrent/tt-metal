# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral vision sequence chunk size for L1-bound matmuls (attention + MLP).

from __future__ import annotations

import os

from models.common.utility_functions import nearest_32


def pixtral_vision_seq_chunk_len(configuration) -> int:
    """L1 matmul seq chunk (env PIXTRAL_VISION_MM_SEQ_CHUNK or min(cfg, cap))."""
    force = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK")
    if force is not None and str(force).strip() != "":
        chunk = max(32, nearest_32(int(force)))
    else:
        cap_raw = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK_CAP", "512")
        cap = max(32, nearest_32(int(cap_raw)))
        cfg_chunk = getattr(configuration, "VISION_MAX_MM_SEQ", cap)
        if cfg_chunk is None:
            cfg_chunk = cap
        chunk = max(32, min(int(cfg_chunk), int(cap)))

    return chunk
