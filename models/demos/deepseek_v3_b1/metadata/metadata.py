# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields

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
        # More advanced serialization could be added here for mixed types to align with data struct on device
        return [getattr(self, f.name) for f in fields(self)]


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
