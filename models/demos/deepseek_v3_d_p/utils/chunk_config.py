# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.

The chunk width is selected with the ``PREFILL_CHUNK_SIZE`` environment variable and defaults to
``DEFAULT_PREFILL_CHUNK_TOKENS``. It must be a multiple of ``PREFILL_SP_FACTOR * 32`` so every chip's slice is a
whole number of tiles. Every consumer (runner, producer, KV chunk table, tests) reads it from here, so exporting
the variable once, identically for the runner and the producer, is all a deployment has to do.
"""

import os

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
DEFAULT_PREFILL_CHUNK_TOKENS = 5 * 1024  # Production configuration across prefill models
PREFILL_CHUNK_SIZE_ENV = "PREFILL_CHUNK_SIZE"


def prefill_chunk_tokens() -> int:
    """The chunk width in tokens: ``$PREFILL_CHUNK_SIZE``, else the production default. Read at call time, so it
    also sees values a launcher or manifest exports after this module was imported."""
    return int(os.environ.get(PREFILL_CHUNK_SIZE_ENV) or DEFAULT_PREFILL_CHUNK_TOKENS)


PREFILL_CHUNK_TOKENS = prefill_chunk_tokens()
PREFILL_CHUNK_TOKENS_PER_CHIP = PREFILL_CHUNK_TOKENS // PREFILL_SP_FACTOR
