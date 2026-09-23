# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only sequence geometry for the fixed SP4/TP8 Llama prefill layout."""

from dataclasses import dataclass
from typing import ClassVar

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model

DEFAULT_MAX_SEQ_LEN = 2048
MAX_CONTEXT_LENGTH = Model.MAX_POSITION_EMBEDDINGS
CHUNK_SIZE = 1024
CACHE_PAGE_SIZE = 32
DEFAULT_NUM_USERS = 2


@dataclass(frozen=True)
class PrefillGeometry:
    """Allocated capacity; logical chunks may end at any token within this limit.

    ``num_users`` is the number of concurrently cached sequences (KV slots). It sets only the packed
    cache's batch extent, so raising it costs DRAM and nothing else; the gather addresses a slot by
    index, and every other dimension here is per-slot.
    """

    # Fixed execution layout. Only the allocated context capacity and slot count vary between
    # instances.
    mesh_shape: ClassVar[tuple[int, int]] = (4, 8)
    sp_axis: ClassVar[int] = 0
    tp_axis: ClassVar[int] = 1
    chunk_size: ClassVar[int] = CHUNK_SIZE
    cache_page_size: ClassVar[int] = CACHE_PAGE_SIZE

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
    def sp(self):
        return self.mesh_shape[self.sp_axis]

    @property
    def tp(self):
        return self.mesh_shape[self.tp_axis]

    @property
    def num_devices(self):
        return self.sp * self.tp

    @property
    def local_sequence(self):
        return self.chunk_size // self.sp

    @property
    def local_q_heads(self):
        return Model.NUM_ATTENTION_HEADS // self.tp

    @property
    def local_kv_heads(self):
        return Model.NUM_KEY_VALUE_HEADS // self.tp

    @property
    def local_cache_sequence(self):
        return self.max_seq_len // self.sp

    @property
    def cache_shape(self):
        return (self.num_users * Model.NUM_LAYERS, self.local_kv_heads, self.local_cache_sequence, Model.HEAD_DIM)

    @property
    def rope_local_sequence(self):
        # Indexed RoPE reads the complete physical chunk even when the final logical tail is short.
        return (self.max_seq_len + self.chunk_size) // self.sp

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
        return min(((populated + self.chunk_size - 1) // self.chunk_size) * self.chunk_size, self.max_seq_len)

    def prefix_gather_block_order(self, extent):
        """Gathered-buffer block indices that restore natural token order across ``extent`` tokens.

        A partial gather transfers fewer source pages but does NOT compact its destination: every rank
        keeps its full-capacity output slot so cache offsets stay stable. The stride below therefore
        stays keyed to max_seq_len, and only the block COUNT shrinks with the prefix.
        """
        if extent % self.local_sequence:
            raise ValueError(f"extent must be a multiple of {self.local_sequence}, got {extent}")
        blocks_per_rank = self.max_seq_len // self.chunk_size
        return tuple(
            (block % self.sp) * blocks_per_rank + block // self.sp for block in range(extent // self.local_sequence)
        )

    @property
    def gather_block_order(self):
        return self.prefix_gather_block_order(self.max_seq_len)

    def validate_cache_metadata(self, cache):
        names = ("num_users", "num_layers", "max_seq_len", "sp")
        actual = tuple(getattr(cache, name, None) for name in names)
        expected = (self.num_users, Model.NUM_LAYERS, self.max_seq_len, self.sp)
        if any(type(value) is not int for value in actual) or actual != expected:
            raise ValueError(
                f"Llama KV cache metadata must be num_users={self.num_users}, num_layers={Model.NUM_LAYERS}, "
                f"max_seq_len={self.max_seq_len}, sp={self.sp}; got {actual}"
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


# Share the fixed layout without importing TTNN. Capacity-dependent callers construct their own geometry.
PREFILL_LAYOUT = PrefillGeometry()
SP_BLOCK_SIZE = PREFILL_LAYOUT.local_sequence


def validate_mesh(mesh_device, mesh_config, owner):
    """Reject unsupported placement before a module allocates tensors or dispatches work."""
    layout = PREFILL_LAYOUT
    required = ("mesh_shape", "tp", "tp_axis", "sp_axis", "sp")
    missing = [name for name in required if not hasattr(mesh_config, name)]
    if missing:
        raise ValueError(f"{owner} mesh_config is missing: {', '.join(missing)}")
    if tuple(mesh_config.mesh_shape) != layout.mesh_shape:
        raise ValueError(f"{owner} requires mesh_shape={layout.mesh_shape}, got {tuple(mesh_config.mesh_shape)}")
    if (mesh_config.sp, mesh_config.tp, mesh_config.sp_axis, mesh_config.tp_axis) != (
        layout.sp,
        layout.tp,
        layout.sp_axis,
        layout.tp_axis,
    ):
        raise ValueError(
            f"{owner} requires SP={layout.sp} on mesh axis {layout.sp_axis} and TP={layout.tp} on mesh axis {layout.tp_axis}; "
            f"got SP={mesh_config.sp}, TP={mesh_config.tp}, "
            f"sp_axis={mesh_config.sp_axis}, tp_axis={mesh_config.tp_axis}"
        )
    if tuple(mesh_device.shape) != layout.mesh_shape or mesh_device.get_num_devices() != layout.num_devices:
        raise ValueError(
            f"{owner} device requires {layout.mesh_shape} with {layout.num_devices} chips; "
            f"got shape={tuple(mesh_device.shape)}, devices={mesh_device.get_num_devices()}"
        )
