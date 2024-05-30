# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtFalconEmbeddings(torch.nn.Module):
    def __init__(self, devices, state_dict, cache_path, model_config):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.model_config = model_config
        self.num_devices = len(devices)

        base_name = "transformer.word_embeddings.weight"
        torch_weights = [
            weight.unsqueeze(0).unsqueeze(0)
            for weight in torch.chunk(self.state_dict[base_name], self.num_devices, dim=-1)
        ]

        cache_name = lambda name, device_id: cache_path / (f"{name}_{device_id}_{self.num_devices}")
        as_tensor = lambda tensor, name, device_id: ttnn.as_tensor(
            tensor,
            dtype=ttnn.bfloat16,  # row_major has to be bfloat16 for now
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.devices[device_id],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name, device_id),
        )

        self.embd_weights = [
            as_tensor(torch_weight, base_name, device_id) for device_id, torch_weight in enumerate(torch_weights)
        ]

    def set_model_config(self, model_config):
        self.model_config = model_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_devices):
            x[i] = ttnn.experimental.tensor.embeddings(
                x[i], self.embd_weights[i], tilized=True, output_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"]
            )

        if self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"].is_sharded():
            for i in range(self.num_devices):
                x[i] = ttnn.experimental.tensor.interleaved_to_sharded(
                    x[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
                )

        return x
