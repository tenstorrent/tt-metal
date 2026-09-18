# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only sequence geometry for the fixed SP4/TP8 Llama prefill layout."""

from dataclasses import dataclass

DEFAULT_MAX_SEQ_LEN = 2048
MAX_CONTEXT_LENGTH = 131072
CHUNK_SIZE = 1024
SP_BLOCK_SIZE = 256
CACHE_PAGE_SIZE = 32
NUM_LAYERS = 32
DEFAULT_NUM_USERS = 2


@dataclass(frozen=True)
class PrefillGeometry:
    """Allocated capacity; logical chunks may end at any token within this limit.

    ``num_users`` is the number of concurrently cached sequences (KV slots). It sets only the packed
    cache's batch extent, so raising it costs DRAM and nothing else; the gather addresses a slot by
    index, and every other dimension here is per-slot. Measured: 2 slots and 1681 slots prefill at
    the same 151 ms per chunk (docs/kv-slot-capacity.md), so the count is a memory decision alone.
    """

    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    num_users: int = DEFAULT_NUM_USERS

    def __post_init__(self):
        if type(self.max_seq_len) is not int:
            raise TypeError("max_seq_len must be an eager Python int")
        if not 0 < self.max_seq_len <= MAX_CONTEXT_LENGTH or self.max_seq_len % CHUNK_SIZE:
            raise ValueError("max_seq_len must be a positive multiple of 1024 through 131072")
        if type(self.num_users) is not int:
            raise TypeError("num_users must be an eager Python int")
        if self.num_users < 1:
            raise ValueError(f"num_users must be a positive int, got {self.num_users}")

    @property
    def local_cache_sequence(self):
        return self.max_seq_len // 4

    @property
    def cache_shape(self):
        return (self.num_users * NUM_LAYERS, 1, self.local_cache_sequence, 128)

    @property
    def rope_local_sequence(self):
        # Indexed RoPE reads the complete physical chunk even when the final logical tail is short.
        return (self.max_seq_len + CHUNK_SIZE) // 4

    def gathered_prefix_extent(self, populated):
        """Round a populated prefix up to whole block-cyclic chunks, clamped to capacity.

        Block-cyclic storage is only addressable in complete chunks: token ``p`` lives on SP rank
        ``(p % 1024) // 256``, so covering any token of a chunk means covering all four ranks' blocks
        for that chunk. This is the active extent to hand to the gather.
        """
        if type(populated) is not int:
            raise TypeError("populated must be an eager Python int")
        if not 0 <= populated <= self.max_seq_len:
            raise ValueError(f"populated must lie in [0, max_seq_len={self.max_seq_len}], got {populated}")
        return min(((populated + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE, self.max_seq_len)

    def prefix_gather_block_order(self, extent):
        """Gathered-buffer block indices that restore natural token order across ``extent`` tokens.

        A partial gather transfers fewer source pages but does NOT compact its destination: every rank
        keeps its full-capacity output slot so cache offsets stay stable. The stride below therefore
        stays keyed to max_seq_len, and only the block COUNT shrinks with the prefix.
        """
        if extent % SP_BLOCK_SIZE:
            raise ValueError(f"extent must be a multiple of {SP_BLOCK_SIZE}, got {extent}")
        blocks_per_rank = self.max_seq_len // CHUNK_SIZE
        return tuple((block % 4) * blocks_per_rank + block // 4 for block in range(extent // SP_BLOCK_SIZE))

    @property
    def gather_block_order(self):
        return self.prefix_gather_block_order(self.max_seq_len)

    def validate_cache_metadata(self, cache):
        names = ("num_users", "num_layers", "max_seq_len", "sp")
        actual = tuple(getattr(cache, name, None) for name in names)
        expected = (self.num_users, NUM_LAYERS, self.max_seq_len, 4)
        if any(type(value) is not int for value in actual) or actual != expected:
            raise ValueError(
                f"Llama KV cache metadata must be num_users={self.num_users}, num_layers={NUM_LAYERS}, "
                f"max_seq_len={self.max_seq_len}, sp=4; got {actual}"
            )

    def validate_chunk_range(self, actual_start, actual_end, *, allow_empty=False):
        if type(actual_start) is not int or type(actual_end) is not int:
            raise TypeError("actual_start and actual_end must be eager Python ints")
        if actual_start < 0 or actual_start % CACHE_PAGE_SIZE:
            raise ValueError("actual_start must be nonnegative and aligned to 32 tokens")
        minimum_end = actual_start if allow_empty else actual_start + 1
        if not minimum_end <= actual_end <= self.max_seq_len:
            raise ValueError(
                f"actual range must stay within max_seq_len={self.max_seq_len}; " f"got [{actual_start}, {actual_end})"
            )
        if actual_end - actual_start > CHUNK_SIZE:
            raise ValueError("a chunk can contain at most 1024 valid tokens")
