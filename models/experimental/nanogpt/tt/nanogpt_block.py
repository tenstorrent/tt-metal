# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import tt_lib
import models.experimental.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
import models.experimental.nanogpt.tt.nanogpt_attention as nanogpt_attention


class TtBlock(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()

        self.device = device
        self.config = config

        self.beta_1 = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".ln_1.bias" + str(dtype) + ".bin")

        self.gamma_1 = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".ln_1.weight" + str(dtype) + ".bin")

        self.ln_1 = tt_lib.tensor.layernorm

        self.attn = nanogpt_attention.TtCausalSelfAttention(
            config, f"{base_address}.attn", device, tt_cache_path, dtype
        )

        self.beta_2 = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".ln_2.bias" + str(dtype) + ".bin")

        self.gamma_2 = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".ln_2.weight" + str(dtype) + ".bin")

        self.ln_2 = tt_lib.tensor.layernorm

        self.mlp = nanogpt_mlp.TtMLP(f"{base_address}.mlp", self.config, device, tt_cache_path, dtype)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        tmp = self.attn.forward(self.ln_1(x, eps=1e-5, gamma=self.gamma_1, beta=self.beta_1))
        x = tt_lib.tensor.add(x, tmp)

        tmp = self.mlp.forward(self.ln_2(x, eps=1e-5, gamma=self.gamma_2, beta=self.beta_2))
        x = tt_lib.tensor.add(x, tmp)

        return x
