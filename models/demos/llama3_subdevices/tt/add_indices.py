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
        self.tt_indices_device_offsets_uint16 = ttnn.from_torch(
            self.indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.tt_indices_device_offsets_int32 = ttnn.from_torch(
            self.indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.tt_indices_device_offsets_uint32 = ttnn.from_torch(
            self.indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        torch_output_tensor = torch.zeros(1, 1, 32, num_local_top_k * self.args.cluster_shape[0], dtype=torch.int64)
        self.typecast_output = ttnn.from_torch(
            torch_output_tensor,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor):
        # topk_indices_gathered_sharded_int32 = ttnn.typecast(x, dtype=ttnn.int32, output_tensor=self.typecast_output)
        # topk_indices_gathered_sharded_uint32 = ttnn.to_memory_config(x, self.args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"], dtype=ttnn.uint32)
        topk_indices_gathered_sharded_int32 = ttnn.to_memory_config(
            x, self.args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"], dtype=ttnn.int32
        )
        ttnn.deallocate(x)
        # topk_global_indices_uint16 = ttnn.add(self.tt_indices_device_offsets_uint16, x, dtype=ttnn.uint16)
        topk_global_indices_int32 = ttnn.add(
            self.tt_indices_device_offsets_int32, topk_indices_gathered_sharded_int32, dtype=ttnn.int32
        )
        # topk_global_indices_uint32 = ttnn.add(self.tt_indices_device_offsets_uint32, topk_indices_gathered_sharded_uint32, dtype=ttnn.uint32)
        breakpoint()
        # ttnn.deallocate(topk_indices_gathered_sharded)

        return topk_global_indices_int32
