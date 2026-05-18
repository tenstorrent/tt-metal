# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Shared Pixtral vision sequence chunk size for L1-bound matmuls (attention + MLP).

from __future__ import annotations

import os

from models.common.utility_functions import nearest_32


def pixtral_vision_seq_chunk_len(configuration) -> int:
    """Vision sequence chunk length for L1-heavy matmuls (attention / MLP).

    Env ``PIXTRAL_VISION_MM_SEQ_CHUNK`` forces a value; else ``min(VISION_MAX_MM_SEQ, cap)`` with cap from ``PIXTRAL_VISION_MM_SEQ_CHUNK_CAP`` (default 448).
    """
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

    return chunk
