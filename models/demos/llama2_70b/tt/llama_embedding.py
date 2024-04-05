# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.demos.llama2_70b.tt.llama_common import get_weight_cache_path


class TtLlamaEmbedding(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        base_name = "tok_embeddings.weight"
        torch_weights = torch.chunk(self.state_dict[base_name], self.num_devices, dim=-1)

        as_tensor = lambda tensor, name, device_id: ttnn.as_tensor(
            tensor,
            dtype=ttnn.bfloat16,  # row_major has to be bfloat16 for now
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.devices[device_id],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_weight_cache_path(cache_path, name, device_id, self.num_devices),
        )

        self.embd_weights = [
            as_tensor(torch_weight, base_name, device_id) for device_id, torch_weight in enumerate(torch_weights)
        ]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_devices):
            x[i] = ttnn.embedding(x[i], self.embd_weights[i])

        return x
