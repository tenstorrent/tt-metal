# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
import models.experimental.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
import models.experimental.nanogpt.tt.nanogpt_attention as nanogpt_attention


class TtBlock(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()

        self.device = device
        self.config = config

        self.beta_1 = ttnn.load_tensor(tt_cache_path + base_address + ".ln_1.bias" + str(dtype) + ".bin")

        self.gamma_1 = ttnn.load_tensor(tt_cache_path + base_address + ".ln_1.weight" + str(dtype) + ".bin")

        self.ln_1 = ttnn.layer_norm

        self.attn = nanogpt_attention.TtCausalSelfAttention(
            config, f"{base_address}.attn", device, tt_cache_path, dtype
        )

        self.beta_2 = ttnn.load_tensor(tt_cache_path + base_address + ".ln_2.bias" + str(dtype) + ".bin")

        self.gamma_2 = ttnn.load_tensor(tt_cache_path + base_address + ".ln_2.weight" + str(dtype) + ".bin")

        self.ln_2 = ttnn.layer_norm

        self.mlp = nanogpt_mlp.TtMLP(f"{base_address}.mlp", self.config, device, tt_cache_path, dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        tmp = self.attn.forward(self.ln_1(x, epsilon=1e-5, weight=self.gamma_1, bias=self.beta_1))
        x = ttnn.add(x, tmp)

        tmp = self.mlp.forward(self.ln_2(x, epsilon=1e-5, weight=self.gamma_2, bias=self.beta_2))
        x = ttnn.add(x, tmp)

        return x
