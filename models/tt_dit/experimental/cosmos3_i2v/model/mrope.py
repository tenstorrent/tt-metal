# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unified 3D mRoPE notes.

Phase 1 (tt-symbiote MVP): mRoPE math runs on PyTorch (host) inside HF's
`Cosmos3OmniTransformer`. No tt-symbiote replacement needed because the
mRoPE compute is element-wise on Q/K and is not a heavy op compared to
the matmuls.

Phase 2 (native): implement on device. Two options ordered by preference:

  1. Extend `ttnn.experimental.rotary_embedding_llama` to accept multi-band
     frequencies (preferred — single kernel, ~2 weeks kernel-team work).
  2. Stage three back-to-back `rotary_embedding` calls and rely on op fusion
     (interim, works immediately).

Configuration from transformer/config.json:

    rope_scaling = {
        "rope_type": "default",
        "mrope_section": [24, 20, 20],   # (temporal, height, width)
        "mrope_interleaved": True,
    }
    rope_theta = 5_000_000
    unified_3d_mrope_reset_spatial_ids = True
    unified_3d_mrope_temporal_modality_margin = 15000

Per-head dim is 128. Sections cover 64 of 128 (sin+cos doubles to 128).
"""

from __future__ import annotations
