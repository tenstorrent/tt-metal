# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields

import torch

import ttnn


@dataclass
class DeepseekMetadata:
    FIELD_SIZE_BYTES = 4  # Each field is uint32_t

    position_id: int = 0
    slot_id: int = 0

    @classmethod
    def aligned_size_bytes(cls) -> int:
        alignment = 64
        unpadded_size = len(fields(cls)) * cls.FIELD_SIZE_BYTES
        return (unpadded_size + alignment - 1) // alignment * alignment

    def to_list(self) -> list[int]:
        # More advanced serialization could be added here for mixed types to align with data struct on device
        return [self.position_id, self.slot_id]


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
