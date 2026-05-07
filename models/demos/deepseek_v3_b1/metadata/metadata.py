# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass, field

import torch

import ttnn

MAX_SPECULATIVE_TOKENS = 4
MAX_WINDOW_TOKENS = MAX_SPECULATIVE_TOKENS + 1
TOPK_METADATA_COUNT = 15
RELAXED_ACCEPT_TOPN = TOPK_METADATA_COUNT
METADATA_SCORE_WORDS = (TOPK_METADATA_COUNT + 1) // 2
METADATA_PAGE_WORDS = 64
METADATA_TENSOR_BYTES = METADATA_PAGE_WORDS * 4
METADATA_TENSOR_NUM_BF16 = METADATA_TENSOR_BYTES // 2
METADATA_TENSOR_NUM_UINT32 = METADATA_TENSOR_BYTES // 4


def _f32_bits(value: float) -> int:
    """IEEE-754 fp32 bit pattern as a uint32 (little-endian host)."""
    return int.from_bytes(struct.pack("<f", float(value)), byteorder="little")


@dataclass
class DeepseekMetadata:
    FIELD_SIZE_BYTES = 4  # Each field is uint32_t
    MAX_SPECULATIVE_TOKENS = MAX_SPECULATIVE_TOKENS
    MAX_WINDOW_TOKENS = MAX_WINDOW_TOKENS
    TOPK_METADATA_COUNT = TOPK_METADATA_COUNT
    RELAXED_ACCEPT_TOPN = RELAXED_ACCEPT_TOPN
    PAGE_SIZE_WORDS = METADATA_PAGE_WORDS

    # Fixed metadata page layout, one uint32 per scalar field:
    #   [0] token_type
    #   [1] request_id
    #   [2] token_id
    #   [3] position_id
    #   [4] lane_idx
    #   [5] temperature
    #   [6] top_k
    #   [7] top_p
    #   [8:13] candidate_token_ids
    #   [13:17] prefill_token_ids
    #   [17:32] p_top15_indices
    #   [32:40] p_top15_scores, two bf16/uint16 scores per uint32 word
    #   [40:55] q_top15_indices
    #   [55:63] q_top15_scores, two bf16/uint16 scores per uint32 word
    token_type: int = 0
    request_id: int = 0
    token_id: int = 0
    position_id: int = 0
    lane_idx: int = 0
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    candidate_token_ids: list[int] = field(default_factory=list)
    prefill_token_ids: list[int] = field(default_factory=list)
    p_top15_indices: list[int] = field(default_factory=list)
    p_top15_scores: list[int] = field(default_factory=list)
    q_top15_indices: list[int] = field(default_factory=list)
    q_top15_scores: list[int] = field(default_factory=list)

    @classmethod
    def aligned_size_bytes(cls) -> int:
        return METADATA_TENSOR_BYTES

    @staticmethod
    def _pack_u16_pairs(values: list[int], count: int) -> list[int]:
        values = [int(value) & 0xFFFF for value in values[:count]]
        values += [0] * (count - len(values))
        if len(values) % 2:
            values.append(0)
        return [values[idx] | (values[idx + 1] << 16) for idx in range(0, len(values), 2)]

    def to_list(self) -> list[int]:
        candidate_token_ids = [int(value) for value in self.candidate_token_ids[: self.MAX_WINDOW_TOKENS]]
        prefill_token_ids = [int(value) for value in self.prefill_token_ids[: self.MAX_SPECULATIVE_TOKENS]]
        p_top15_indices = [int(value) for value in self.p_top15_indices[: self.TOPK_METADATA_COUNT]]
        q_top15_indices = [int(value) for value in self.q_top15_indices[: self.TOPK_METADATA_COUNT]]

        candidate_token_ids += [0] * (self.MAX_WINDOW_TOKENS - len(candidate_token_ids))
        prefill_token_ids += [0] * (self.MAX_SPECULATIVE_TOKENS - len(prefill_token_ids))
        p_top15_indices += [0] * (self.TOPK_METADATA_COUNT - len(p_top15_indices))
        q_top15_indices += [0] * (self.TOPK_METADATA_COUNT - len(q_top15_indices))
        p_top15_scores = self._pack_u16_pairs(self.p_top15_scores, self.TOPK_METADATA_COUNT)
        q_top15_scores = self._pack_u16_pairs(self.q_top15_scores, self.TOPK_METADATA_COUNT)

        values = [
            int(self.token_type),
            int(self.request_id),
            int(self.token_id),
            int(self.position_id),
            int(self.lane_idx),
            _f32_bits(self.temperature),
            int(self.top_k),
            _f32_bits(self.top_p),
            *candidate_token_ids,
            *prefill_token_ids,
            *p_top15_indices,
            *p_top15_scores,
            *q_top15_indices,
            *q_top15_scores,
        ]
        if len(values) > self.PAGE_SIZE_WORDS:
            raise ValueError(f"DeepseekMetadata has {len(values)} fields, exceeding {self.PAGE_SIZE_WORDS} words")
        return values + [0] * (self.PAGE_SIZE_WORDS - len(values))


def create_metadata_tensor(
    mesh_device: ttnn.MeshDevice, grid: ttnn.CoreRangeSet, metadata: DeepseekMetadata
) -> ttnn.Tensor:
    torch_metadata = torch.tensor(metadata.to_list(), dtype=torch.uint32).repeat(grid.num_cores(), 1)
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
