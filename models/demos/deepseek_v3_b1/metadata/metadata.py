# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

import ttnn


@dataclass
class DeepseekMetadata:
    FIELD_SIZE_BYTES = 4  # Each field is uint32_t
    MAX_SPECULATIVE_TOKENS = 4
    MAX_WINDOW_TOKENS = MAX_SPECULATIVE_TOKENS + 1
    RELAXED_ACCEPT_TOPN = 10
    PAGE_SIZE_WORDS = 64

    # Fixed metadata page layout, one uint32 per scalar field:
    #   [0] token_type
    #   [1] slot_id
    #   [2] token_id
    #   [3] position_id
    #   [4] prefill_token_id
    #   [5] lane_idx
    #   [6] window_start_pos
    #   [7] num_window_tokens
    #   [8:13] candidate_token_ids
    #   [13:18] candidate_positions
    #   [18] target_topn_count
    #   [19:29] target_topn_tokens
    #   [29:39] target_topn_probs
    token_type: int = 0
    slot_id: int = 0
    token_id: int = 0
    position_id: int = 0
    prefill_token_id: int = 0
    lane_idx: int = 0
    window_start_pos: int = 0
    num_window_tokens: int = 0
    candidate_token_ids: list[int] = field(default_factory=list)
    candidate_positions: list[int] = field(default_factory=list)
    target_topn_count: int = 0
    target_topn_tokens: list[int] = field(default_factory=list)
    target_topn_probs: list[int] = field(default_factory=list)

    @classmethod
    def aligned_size_bytes(cls) -> int:
        return cls.PAGE_SIZE_WORDS * cls.FIELD_SIZE_BYTES

    def to_list(self) -> list[int]:
        candidate_token_ids = [int(value) for value in self.candidate_token_ids[: self.MAX_WINDOW_TOKENS]]
        candidate_positions = [int(value) for value in self.candidate_positions[: self.MAX_WINDOW_TOKENS]]
        target_topn_tokens = [int(value) for value in self.target_topn_tokens[: self.RELAXED_ACCEPT_TOPN]]
        target_topn_probs = [int(value) for value in self.target_topn_probs[: self.RELAXED_ACCEPT_TOPN]]

        candidate_token_ids += [0] * (self.MAX_WINDOW_TOKENS - len(candidate_token_ids))
        candidate_positions += [0] * (self.MAX_WINDOW_TOKENS - len(candidate_positions))
        target_topn_tokens += [0] * (self.RELAXED_ACCEPT_TOPN - len(target_topn_tokens))
        target_topn_probs += [0] * (self.RELAXED_ACCEPT_TOPN - len(target_topn_probs))

        values = [
            int(self.token_type),
            int(self.slot_id),
            int(self.token_id),
            int(self.position_id),
            int(self.prefill_token_id),
            int(self.lane_idx),
            int(self.window_start_pos),
            int(self.num_window_tokens),
            *candidate_token_ids,
            *candidate_positions,
            int(self.target_topn_count),
            *target_topn_tokens,
            *target_topn_probs,
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
