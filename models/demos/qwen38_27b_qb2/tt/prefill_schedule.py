# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bound transient batched-prefill rows without reducing resident context."""

DEFAULT_MAX_BATCH_TOKENS = 32 * 4096


def prefill_chunk_size(batch, max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS):
    """Return an aligned per-user chunk under the aggregate activation budget.

    The default preserves existing chunks for every supported decode batch.
    A 32,768-token budget uses 4,096/2,048/1,024-token chunks at B8/B16/B32.
    At least 4,096 rows keep the existing single-user prefill/trace contract.
    This bounds chunk geometry, not exact allocator bytes or total KV capacity.
    """
    if type(batch) is not int or not 1 <= batch <= 32:
        raise ValueError("Prefill chunk planning requires 1..32 users")
    if type(max_batch_tokens) is not int or max_batch_tokens < 4096 or max_batch_tokens % 32:
        raise ValueError("Prefill batch-token budget must be a multiple of 32 and at least 4096")
    return min(4096, max_batch_tokens // batch // 32 * 32)
