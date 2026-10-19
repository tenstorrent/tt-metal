# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bounded KV rings for the sliding-window attention layers.

A sliding layer attends to the last 128 positions, so it keeps only
SLIDING_RING_TOKENS positions per device slot: the paged-cache kernels take
``cache_position_modulo`` and address position ``p`` at ring slot
``p mod SLIDING_RING_TOKENS`` through the first ring blocks of the layer's page
table. The ring must span at least two decode K chunks (128 tokens each) so the
two chunks the window can touch never alias the same physical tiles, and it must
still hold the 128 positions before any prefill resume point: a resume is aligned
down to 512 from a block-aligned cached length, so the ring keeps 512 + 128
positions behind the last one written, rounded up to whole K chunks (768).
"""

import os

PAGE_SIZE = 64
PREFILL_CHUNK_ALIGN = 512
DECODE_K_CHUNK = 128
ENV_SLIDING_RING = "GPT_OSS_120B_SLIDING_RING"
ENV_SLIDING_RING_TOKENS = "GPT_OSS_120B_SLIDING_RING_TOKENS"
MIN_SLIDING_RING_TOKENS = 768


def _ring_tokens() -> int:
    value = int(os.environ.get(ENV_SLIDING_RING_TOKENS, "768"))
    if value < MIN_SLIDING_RING_TOKENS or value % DECODE_K_CHUNK:
        raise ValueError(
            f"{ENV_SLIDING_RING_TOKENS} must be a multiple of {DECODE_K_CHUNK} and at least "
            f"{MIN_SLIDING_RING_TOKENS} to preserve the window across prefix resumes, got {value}"
        )
    return value


SLIDING_RING_TOKENS = _ring_tokens()
SLIDING_RING_BLOCKS = SLIDING_RING_TOKENS // PAGE_SIZE


def sliding_ring_enabled() -> bool:
    return os.environ.get(ENV_SLIDING_RING, "1") != "0"


def ring_modulo_for_layer(layer_type: str) -> int | None:
    if layer_type == "sliding_attention" and sliding_ring_enabled():
        return SLIDING_RING_TOKENS
    return None
