# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh


class TtLlamaEmbedding:
    def __init__(
        self,
        mesh_device,
        state_dict,
        cache_path,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()

        base_name = "tok_embeddings.weight"
        # torch_weights = [
        #     weight.unsqueeze(0).unsqueeze(0)
        #     for weight in torch.chunk(self.state_dict[base_name], self.num_devices, dim=-1)
        # ]

        # cache_name = lambda name, device_id: cache_path / (f"{name}_{device_id}_{self.num_devices}")
        # as_tensor = lambda tensor, name, device_id: ttnn.as_tensor(
        #     tensor,
        #     dtype=ttnn.bfloat16,  # row_major has to be bfloat16 for now
        #     layout=ttnn.ROW_MAJOR_LAYOUT,
        #     device=self.devices[device_id],
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=cache_name(name, device_id),
        # )

        # self.embd_weights = [
        #     as_tensor(torch_weight, base_name, device_id) for device_id, torch_weight in enumerate(torch_weights)
        # ]

        embd_weights_ttn = ttnn.as_tensor(
            self.state_dict[base_name].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,  # row_major has to be bfloat16 for now
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
            cache_file_name=cache_path / base_name,
        )
        self.emb_weights = ttnn.to_device(embd_weights_ttn, mesh_device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(
            x, self.emb_weights, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])

        return x
