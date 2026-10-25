# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
PREFILL_CHUNK_TOKENS = 5 * 1024  # Production configuration across prefill models
PREFILL_CHUNK_TOKENS_PER_CHIP = PREFILL_CHUNK_TOKENS // PREFILL_SP_FACTOR


def kv_slab_tokens(chunk_size_global: int) -> int:
    """Global width S of the KV-cache layout slab (GLM_KV_SLAB_TOKENS; default: the compute chunk).

    The block-cyclic KV layout is keyed to S, not to the compute chunk C: SP chip c, local row lr holds position
    (lr // s) * S + c * s + lr % s with s = S / sp, and a chunk of C = k * S tokens writes k slabs (C / sp contiguous
    local rows per chip). With one fixed S (e.g. 1024 on 8x4) a cache written by 5120-token chunks can be extended by
    native 1024 / 2048 chunks. S == C (the default) is exactly today's layout.
    """
    import os

    slab = int(os.environ.get("GLM_KV_SLAB_TOKENS", "0") or 0)
    if slab <= 0 or slab >= chunk_size_global:
        return chunk_size_global
    assert chunk_size_global % slab == 0, f"chunk {chunk_size_global} must be a multiple of GLM_KV_SLAB_TOKENS {slab}"
    return slab
