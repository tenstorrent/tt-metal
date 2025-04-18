# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TTAddIndices(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device

        # Create indices tensor
        num_local_top_k = 32
        self.indices_device_offsets = torch.ones(
            1, 1, 32, num_local_top_k * self.args.cluster_shape[0], dtype=torch.int64
        )
        per_device_vocab_size = self.args.padded_vocab_size // self.args.cluster_shape[0]
        per_device_vocab_size = 5000
        for device_id in range(self.args.cluster_shape[0]):
            self.indices_device_offsets[:, :, :, device_id * num_local_top_k : (device_id + 1) * num_local_top_k] = (
                device_id * per_device_vocab_size
            )
        self.tt_indices_device_offsets_int32 = ttnn.from_torch(
            self.indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )

    def forward(self, x: ttnn.Tensor):
        topk_indices_gathered_sharded = ttnn.typecast(x, dtype=ttnn.uint32, sub_core_grids=self.sub_core_grids)
        topk_indices_gathered_sharded = ttnn.typecast(
            topk_indices_gathered_sharded, dtype=ttnn.int32, sub_core_grids=self.sub_core_grids
        )

        # ttnn.deallocate(x)
        breakpoint()
        topk_global_indices_int32 = ttnn.add(
            self.tt_indices_device_offsets_int32, topk_indices_gathered_sharded, dtype=ttnn.int32
        )
        # ttnn.deallocate(topk_indices_gathered_sharded)

        return topk_global_indices_int32
