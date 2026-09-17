# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-derived chronology and fixed-shape selection records.

All arithmetic lives in the native chronological_topology operation. Python only
selects fixed record locations; it never reads the changing offset.
"""

from dataclasses import dataclass

import torch

import ttnn


def rank_tensor(device: ttnn.MeshDevice, axis: int) -> ttnn.Tensor:
    dims = [None, None]
    dims[axis] = 0
    return ttnn.from_torch(
        torch.arange(tuple(device.shape)[axis], dtype=torch.int32).reshape(-1, 1),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=tuple(dims), mesh_shape=tuple(device.shape)),
    )


@dataclass(frozen=True)
class DeviceChronology:
    controls: ttnn.Tensor
    sp_size: int

    def indices(self, row: int, width: int) -> ttnn.Tensor:
        return ttnn.reshape(
            ttnn.slice(self.controls, (row, 0), (row + 1, width), memory_config=ttnn.L1_MEMORY_CONFIG), (width,)
        )

    def select_block(
        self,
        tensor: ttnn.Tensor,
        row: int,
        parts: int,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        return ttnn.slice(
            tensor,
            self.indices(row, 4),
            self.indices(row + 1, 4),
            slice_dim=0,
            num_devices=parts,
            memory_config=memory_config,
        )

    def select_rows(self, tensor: ttnn.Tensor, row: int) -> ttnn.Tensor:
        width = tensor.shape[-1]
        table = ttnn.reshape(tensor, (-1, width))
        selected = ttnn.embedding(
            self.indices(row, 3), table, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.reshape(selected, (1, 3, width))
