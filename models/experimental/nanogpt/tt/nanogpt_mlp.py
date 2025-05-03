# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.helper_funcs import Linear


class TtMLP(torch.nn.Module):
    def __init__(self, base_address, config, device, tt_cache_path, dtype):
        super().__init__()
        # Get the weights
        self.tt_weight_c_fc = ttnn.load_tensor(tt_cache_path + base_address + ".c_fc.weight" + str(dtype) + ".bin")
        self.tt_weight_c_proj = ttnn.load_tensor(tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".bin")

        self.config = config
        self.device = device

        # Load biases
        self.tt_bias_c_fc = ttnn.load_tensor(tt_cache_path + base_address + ".c_fc.bias" + str(dtype) + ".bin")

        self.tt_bias_c_proj = ttnn.load_tensor(tt_cache_path + base_address + ".c_proj.bias" + str(dtype) + ".bin")

        self.tt_weight_c_fc = ttnn.transpose(self.tt_weight_c_fc, -2, -1)
        self.tt_weight_c_proj = ttnn.transpose(self.tt_weight_c_proj, -2, -1)

        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, self.tt_weight_c_fc, self.tt_bias_c_fc)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, self.tt_weight_c_proj, self.tt_bias_c_proj)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x1 = self.c_fc(x)
        x2 = ttnn.gelu(x1)
        x3 = self.c_proj(x2)

        return x3
