# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass, field

import torch

import ttnn

# Full size of the on-device DeepseekMetadata struct (header + p_indices + p_scores).
# MUST match `sizeof(deepseek_b1_ops::DeepseekMetadata)` in `metadata.hpp`. Used to
# size the LM-head sampling source/destination metadata buffers (the source unicasts
# the whole struct, and the destination has 192 B of trailing space that sampling.hpp
# fills with the post-top-P p_indices / p_scores arrays).
#
# This is intentionally separate from `aligned_size_bytes()` (which describes the
# header-only socket page used by the upstream pipeline) and from
# `TOKEN_META_PAGE_SIZE_BYTES` (which describes the deferred output socket page).
METADATA_TENSOR_BYTES = 256
METADATA_TENSOR_NUM_BF16 = METADATA_TENSOR_BYTES // 2
METADATA_TENSOR_NUM_UINT32 = METADATA_TENSOR_BYTES // 4

# On-device DeepseekMetadata array capacities (must match metadata.hpp).
METADATA_P_INDICES_CAPACITY = 32  # uint32 each → 32 words = 128 B
METADATA_P_SCORES_CAPACITY = 32  # bf16  each → 16 words = 64 B (2 bf16 packed per uint32)


def _f32_bits(value: float) -> int:
    """IEEE-754 fp32 bit pattern as a uint32 (little-endian host)."""
    return int.from_bytes(struct.pack("<f", float(value)), byteorder="little")


def _bf16_bits(value: float) -> int:
    """bf16 bit pattern as a uint16. bf16 = top 16 bits of fp32 (truncation)."""
    return (_f32_bits(value) >> 16) & 0xFFFF


@dataclass
class DeepseekMetadata:
    FIELD_SIZE_BYTES = 4  # Each field is uint32_t

    tok0_id: int = 0
    tok0_type: int = 0
    tok0_pos: int = 0
    tok1_id: int = 0
    tok1_type: int = 0
    tok1_pos: int = 0
    slot_id: int = 0
    token_id: int = 0
    position_id: int = 0
    prefill_token_id: int = 0
    temperature: float = 0.0
    k: int = 0
    probability_mass_threshold: float = 0.0
    _pad0: int = 0
    _pad1: int = 0
    _pad2: int = 0
    p_indices: list[int] = field(default_factory=list)
    p_scores: list[float] = field(default_factory=list)

    @classmethod
    def aligned_size_bytes(cls) -> int:
        # Returns the full on-device struct size (header + p_indices + p_scores).
        # Pipeline stages that forward metadata reserve this many bytes per shard
        # so that the LM-head sampling stage can write `p_indices` / `p_scores`
        # in place. The Python dataclass above only mirrors the header fields;
        # the trailing arrays exist solely on device and are filled by sampling.hpp.
        return METADATA_TENSOR_BYTES

    def to_list(self) -> list[int]:
        # Serialize the dataclass into a list of `uint32` words that mirrors the
        # on-device DeepseekMetadata struct layout from metadata.hpp:
        #
        #   words  0..15 : header
        #     0..12  → 13 scalar fields (floats bit-cast as uint32)
        #     13..15 → _pad0 / _pad1 / _pad2
        #   words 16..47 : p_indices[32]  — one uint32 per index
        #   words 48..63 : p_scores[32]   — 32 bf16 packed two-per-uint32
        #                                  (low halfword → even index, high → odd,
        #                                   matches LE access of uint16_t[32] as uint32_t[16])
        #
        # Total: 64 uint32 words = METADATA_TENSOR_BYTES (256 B).
        words: list[int] = [
            self.tok0_id & 0xFFFFFFFF,
            self.tok0_type & 0xFFFFFFFF,
            self.tok0_pos & 0xFFFFFFFF,
            self.tok1_id & 0xFFFFFFFF,
            self.tok1_type & 0xFFFFFFFF,
            self.tok1_pos & 0xFFFFFFFF,
            self.slot_id & 0xFFFFFFFF,
            self.token_id & 0xFFFFFFFF,
            self.position_id & 0xFFFFFFFF,
            self.prefill_token_id & 0xFFFFFFFF,
            _f32_bits(self.temperature),
            self.k & 0xFFFFFFFF,
            _f32_bits(self.probability_mass_threshold),
            self._pad0 & 0xFFFFFFFF,
            self._pad1 & 0xFFFFFFFF,
            self._pad2 & 0xFFFFFFFF,
        ]
        assert len(words) == 16, "header must occupy exactly 16 uint32 words (64 B)"

        if len(self.p_indices) > METADATA_P_INDICES_CAPACITY:
            raise ValueError(
                f"p_indices length {len(self.p_indices)} exceeds capacity " f"{METADATA_P_INDICES_CAPACITY}"
            )
        p_idx = list(self.p_indices) + [0] * (METADATA_P_INDICES_CAPACITY - len(self.p_indices))
        words.extend(int(v) & 0xFFFFFFFF for v in p_idx)

        if len(self.p_scores) > METADATA_P_SCORES_CAPACITY:
            raise ValueError(f"p_scores length {len(self.p_scores)} exceeds capacity " f"{METADATA_P_SCORES_CAPACITY}")

        p_sc = list(self.p_scores) + [0.0] * (METADATA_P_SCORES_CAPACITY - len(self.p_scores))
        for i in range(0, METADATA_P_SCORES_CAPACITY, 2):
            lo = _bf16_bits(p_sc[i])
            hi = _bf16_bits(p_sc[i + 1])
            words.append(((hi & 0xFFFF) << 16) | (lo & 0xFFFF))

        assert len(words) == METADATA_TENSOR_NUM_UINT32, (
            f"expected {METADATA_TENSOR_NUM_UINT32} uint32 words " f"({METADATA_TENSOR_BYTES} B), got {len(words)}"
        )
        return words


def _metadata_to_padded_uint32(metadata: DeepseekMetadata) -> torch.Tensor:
    """Pack ``metadata`` into a 1-D uint32 tensor of length ``aligned_size_bytes()/4``.

    The bit pattern matches the C++ ``DeepseekMetadata`` struct laid out in L1
    immediately after the activation data, padded with zeros out to
    ``aligned_size_bytes()``. Callers reinterpret/cast this buffer to whatever
    dtype the destination tensor uses (bf16 for input shards, uint32 for the
    standalone metadata tensor, etc.).
    """
    aligned_words = DeepseekMetadata.aligned_size_bytes() // DeepseekMetadata.FIELD_SIZE_BYTES
    metadata_words = metadata.to_list()
    assert len(metadata_words) <= aligned_words, (
        f"DeepseekMetadata serialized to {len(metadata_words)} uint32 words, "
        f"but only {aligned_words} fit in aligned_size_bytes()={DeepseekMetadata.aligned_size_bytes()} B"
    )
    padded_words = metadata_words + [0] * (aligned_words - len(metadata_words))
    return torch.tensor(padded_words, dtype=torch.uint32)


def append_metadata_tail(tensor: torch.Tensor, metadata: DeepseekMetadata) -> torch.Tensor:
    """Return ``tensor`` with ``metadata``'s L1 tail bytes appended along the last dim.

    The decoder/attention input shard reserves ``DeepseekMetadata.aligned_size_bytes()``
    bytes immediately after the activation data. The upstream socket writes that
    region in production; for unit tests without a socket we instead pack the
    same bytes into the padded torch tensor that backs the input shard.

    The metadata bytes are reinterpreted as ``tensor.dtype`` (e.g. bf16 yields
    ``aligned_size_bytes()/2`` trailing elements per row), and broadcast across
    any leading dims of ``tensor``.
    """
    metadata_bytes = DeepseekMetadata.aligned_size_bytes()
    elem_bytes = tensor.element_size()
    assert metadata_bytes % elem_bytes == 0, (
        f"DeepseekMetadata size {metadata_bytes} B is not a multiple of "
        f"tensor element size {elem_bytes} B (dtype={tensor.dtype})"
    )
    tail_elems = metadata_bytes // elem_bytes
    tail = _metadata_to_padded_uint32(metadata).view(tensor.dtype)
    leading_shape = tensor.shape[:-1]
    tail = tail.reshape((1,) * len(leading_shape) + (tail_elems,)).expand(*leading_shape, tail_elems)
    return torch.cat([tensor, tail], dim=-1)


def create_metadata_tensor(
    mesh_device: ttnn.MeshDevice, grid: ttnn.CoreRangeSet, metadata: DeepseekMetadata
) -> ttnn.Tensor:
    # Each shard holds the full on-device DeepseekMetadata struct
    # (METADATA_TENSOR_BYTES = 256 B = METADATA_TENSOR_NUM_UINT32 = 64 uint32 words).
    # `metadata.to_list()` returns exactly that many words, packed to mirror the
    # C++ struct layout (header + p_indices + p_scores).
    words = metadata.to_list()
    torch_metadata = torch.tensor(words, dtype=torch.uint32).repeat(grid.num_cores(), 1)
    assert torch_metadata.shape == (grid.num_cores(), METADATA_TENSOR_NUM_UINT32), (
        f"metadata tensor shape {tuple(torch_metadata.shape)} != " f"({grid.num_cores()}, {METADATA_TENSOR_NUM_UINT32})"
    )
    metadata_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (1, torch_metadata.shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_metadata = ttnn.from_torch(
        torch_metadata,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=metadata_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return ttnn_metadata
