# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass, field

import torch

import ttnn

# Must match sizeof(deepseek_b1_ops::DeepseekMetadata) in metadata.hpp.
METADATA_TENSOR_BYTES = 512
METADATA_TENSOR_NUM_BF16 = METADATA_TENSOR_BYTES // 2
METADATA_TENSOR_NUM_UINT32 = METADATA_TENSOR_BYTES // 4

MAX_MTP_LEVELS = 4
NUM_OUTPUT_TOKENS = 1 + MAX_MTP_LEVELS  # 1 base + up to 4 speculative

# Array capacities — must match metadata.hpp.
METADATA_P_INDICES_CAPACITY = 32
METADATA_P_SCORES_CAPACITY = 32
METADATA_Q_INDICES_CAPACITY = 32
METADATA_Q_SCORES_CAPACITY = 32


def _f32_bits(value: float) -> int:
    """IEEE-754 fp32 bit pattern as a uint32 (little-endian host)."""
    return int.from_bytes(struct.pack("<f", float(value)), byteorder="little")


def _bf16_bits(value: float) -> int:
    """bf16 bit pattern as a uint16. bf16 = top 16 bits of fp32 (truncation)."""
    return (_f32_bits(value) >> 16) & 0xFFFF


@dataclass
class DeepseekMetadata:
    """Python mirror of deepseek_b1_ops::DeepseekMetadata (metadata.hpp).

    Header layout (16 uint32 words = 64 bytes):
        word  0    : lane_id   (0 = base, 1..4 = MTP level)
        word  1    : slot_id
        word  2    : token_id
        word  3    : position_id
        words 4-8  : output_token_ids[5]   (base + 4 spec)
        words 9-12 : prefill_token_ids[4]  (one per MTP level)
        word 13    : temperature  (float)
        word 14    : k            (uint32)
        word 15    : p            (float)

    Followed by:
        words 16-47   : p_indices[32]   (uint32)
        words 48-63   : p_scores[32]    (bf16, two packed per uint32)
        words 64-95   : q_indices[32]   (uint32)
        words 96-111  : q_scores[32]    (bf16, two packed per uint32)
        words 112-127 : padding[16]
    """

    FIELD_SIZE_BYTES = 4
    lane_id: int = 0
    slot_id: int = 0
    token_id: int = 0
    position_id: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    prefill_token_ids: list[int] = field(default_factory=list)
    temperature: float = 0.0
    k: int = 0
    p: float = 0.0
    p_indices: list[int] = field(default_factory=list)
    p_scores: list[float] = field(default_factory=list)
    q_indices: list[int] = field(default_factory=list)
    q_scores: list[float] = field(default_factory=list)

    @classmethod
    def aligned_size_bytes(cls) -> int:
        return METADATA_TENSOR_BYTES

    def to_list(self) -> list[int]:
        """Serialize into uint32 words matching the on-device struct layout."""
        out_tok = list(self.output_token_ids) + [0] * (NUM_OUTPUT_TOKENS - len(self.output_token_ids))
        prefill = list(self.prefill_token_ids) + [0] * (MAX_MTP_LEVELS - len(self.prefill_token_ids))

        words: list[int] = [
            self.lane_id & 0xFFFFFFFF,
            self.slot_id & 0xFFFFFFFF,
            self.token_id & 0xFFFFFFFF,
            self.position_id & 0xFFFFFFFF,
        ]
        words.extend(v & 0xFFFFFFFF for v in out_tok[:NUM_OUTPUT_TOKENS])
        words.extend(v & 0xFFFFFFFF for v in prefill[:MAX_MTP_LEVELS])
        words.append(_f32_bits(self.temperature))
        words.append(self.k & 0xFFFFFFFF)
        words.append(_f32_bits(self.p))
        assert len(words) == 16, f"header must be 16 words, got {len(words)}"

        def _pack_u32_array(values: list[int], capacity: int) -> None:
            padded = list(values) + [0] * (capacity - len(values))
            if len(values) > capacity:
                raise ValueError(f"array length {len(values)} exceeds capacity {capacity}")
            words.extend(int(v) & 0xFFFFFFFF for v in padded[:capacity])

        def _pack_bf16_array(values: list[float], capacity: int) -> None:
            padded = list(values) + [0.0] * (capacity - len(values))
            if len(values) > capacity:
                raise ValueError(f"array length {len(values)} exceeds capacity {capacity}")
            for i in range(0, capacity, 2):
                lo = _bf16_bits(padded[i])
                hi = _bf16_bits(padded[i + 1])
                words.append(((hi & 0xFFFF) << 16) | (lo & 0xFFFF))

        _pack_u32_array(self.p_indices, METADATA_P_INDICES_CAPACITY)
        _pack_bf16_array(self.p_scores, METADATA_P_SCORES_CAPACITY)
        _pack_u32_array(self.q_indices, METADATA_Q_INDICES_CAPACITY)
        _pack_bf16_array(self.q_scores, METADATA_Q_SCORES_CAPACITY)

        # Padding to reach 512 bytes = 128 uint32 words.
        words.extend([0] * 16)

        assert (
            len(words) == METADATA_TENSOR_NUM_UINT32
        ), f"expected {METADATA_TENSOR_NUM_UINT32} words ({METADATA_TENSOR_BYTES} B), got {len(words)}"
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
