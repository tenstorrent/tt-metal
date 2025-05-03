# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMistralEmbedding(torch.nn.Module):
    def __init__(
        self,
        device,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device

        base_name = "tok_embeddings.weight"
        torch_weight = self.state_dict[base_name]
        cache_name = weight_cache_path / base_name
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(x, self.weights)
