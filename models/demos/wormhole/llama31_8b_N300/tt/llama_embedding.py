# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtLlamaEmbedding(torch.nn.Module):
    def __init__(
        self,
        device_mesh,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh

        base_name = "tok_embeddings.weight"
        torch_weight = self.state_dict[base_name]
        cache_name = weight_cache_path / base_name
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=self.device_mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device_mesh),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])
        # x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, 32-x.shape[2]), (0, 0)), value=0)
        # x = ttnn.tilize(x, use_multicore=True)
        return x
