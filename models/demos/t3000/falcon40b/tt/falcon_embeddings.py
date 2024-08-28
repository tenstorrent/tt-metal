# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh


class TtFalconEmbeddings(torch.nn.Module):
    def __init__(self, device_mesh, state_dict, cache_path, model_config):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.model_config = model_config
        self.num_devices = device_mesh.get_num_devices()

        base_name = "transformer.word_embeddings.weight"

        self.embd_weights = ttnn.as_tensor(
            tensor=self.state_dict[base_name],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_path / base_name,
            mesh_mapper=ShardTensorToMesh(device_mesh, dim=-1),
            preprocess=lambda x: x.reshape(1, 1, *x.shape),
        )

    def set_model_config(self, model_config):
        self.model_config = model_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(
            x,
            self.embd_weights,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])

        if self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"].is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])

        return x
