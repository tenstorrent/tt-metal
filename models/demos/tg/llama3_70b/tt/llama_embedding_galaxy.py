# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.t3000.llama2_70b.tt.llama_common import ShardTensor2dMesh


class TtLlamaEmbedding_galaxy:
    def __init__(
        self,
        mesh_device,
        cluster_shape,
        state_dict,
        cache_path,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.cluster_shape = cluster_shape

        base_name = "tok_embeddings.weight"
        embedding_cache_name = "tok_embeddings_galaxy.weight"

        self.emb_weights = ttnn.as_tensor(
            self.state_dict[base_name].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,  # row_major has to be bfloat16 for now
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(3, None), cluster_shape=self.cluster_shape),
            cache_file_name=cache_path / embedding_cache_name,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.emb_weights, layout=ttnn.TILE_LAYOUT)  # , dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])

        return x
