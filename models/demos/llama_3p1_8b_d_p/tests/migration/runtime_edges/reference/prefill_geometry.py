# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only sequence geometry for the fixed SP4/TP8 Llama prefill layout."""

from dataclasses import dataclass

DEFAULT_MAX_SEQ_LEN = 2048
MAX_CONTEXT_LENGTH = 131072
CHUNK_SIZE = 1024
SP_BLOCK_SIZE = 256
CACHE_PAGE_SIZE = 32


@dataclass(frozen=True)
class PrefillGeometry:
    """Allocated capacity; logical chunks may end at any token within this limit."""

    max_seq_len: int = DEFAULT_MAX_SEQ_LEN

    def __post_init__(self):
        if type(self.max_seq_len) is not int:
            raise TypeError("max_seq_len must be an eager Python int")
        if not 0 < self.max_seq_len <= MAX_CONTEXT_LENGTH or self.max_seq_len % CHUNK_SIZE:
            raise ValueError("max_seq_len must be a positive multiple of 1024 through 131072")

    @property
    def local_cache_sequence(self):
        return self.max_seq_len // 4

    @property
    def cache_shape(self):
        return (64, 1, self.local_cache_sequence, 128)

    @property
    def rope_local_sequence(self):
        # Indexed RoPE reads the complete physical chunk even when the final logical tail is short.
        return (self.max_seq_len + CHUNK_SIZE) // 4

    @property
    def gather_block_order(self):
        blocks_per_rank = self.max_seq_len // CHUNK_SIZE
        return tuple((block % 4) * blocks_per_rank + block // 4 for block in range(self.max_seq_len // SP_BLOCK_SIZE))

    def validate_cache_metadata(self, cache):
        names = ("num_users", "num_layers", "max_seq_len", "sp")
        actual = tuple(getattr(cache, name, None) for name in names)
        expected = (2, 32, self.max_seq_len, 4)
        if any(type(value) is not int for value in actual) or actual != expected:
            raise ValueError(
                f"Llama KV cache metadata must be num_users=2, num_layers=32, "
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
