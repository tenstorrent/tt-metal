# SPDX-License-Identifier: Apache-2.0
"""Physical prefill scratch policy; independent of logical context/cache size."""


def physical_prefill_chunk_size(batch_size, scheduler_block_size=None, *, cap=None):
    size = 1024 if batch_size > 16 else 4096
    if scheduler_block_size is not None:
        size = min(size, scheduler_block_size)
    if cap is not None:
        if cap not in ("256", "512", "1024", "2048", "4096"):
            raise ValueError("QWEN38_PREFILL_CHUNK_CAP must be256/512/1024/2048/4096")
        size = min(size, int(cap))
    return size
