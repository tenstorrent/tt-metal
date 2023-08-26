import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
import python_api_testing.models.nanogpt.tt.nanogpt_attention as nanogpt_attention
from python_api_testing.models.nanogpt.tt.nanogpt_config import GPTConfig

from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass
import math

from transformers import GPT2LMHeadModel

from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

class TtBlock(nn.Module):
    def __init__(self, config: GPTConfig(), state_dict, base_address, device):
        super().__init__()

        self.device = device
        self.config = config

        self.beta_1 = torch2tt_tensor(
            state_dict[f"{base_address}.ln_1.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.gamma_1 = torch2tt_tensor(
            state_dict[f"{base_address}.ln_1.weight"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.ln_1 = fallback_ops.LayerNorm(
            self.gamma_1,
            self.beta_1,
            eps=1e-5,
            normalized_shape=config.n_embd,
        )

        self.attn = nanogpt_attention.TtCausalSelfAttention(
            config, state_dict, f"{base_address}.attn", device
        )

        self.beta_2 = torch2tt_tensor(
            state_dict[f"{base_address}.ln_2.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.gamma_2 = torch2tt_tensor(
            state_dict[f"{base_address}.ln_2.weight"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.ln_2 = fallback_ops.LayerNorm(
            self.gamma_2, self.beta_2, eps=1e-5, normalized_shape=config.n_embd
        )

        self.mlp = nanogpt_mlp.TtMLP(f"{base_address}.mlp", self.config, state_dict, device)

    def forward(self, x):
        tmp = self.attn.forward(self.ln_1(x))
        x = tt_lib.tensor.add(x, tmp)

        tmp = self.mlp.forward(self.ln_2(x))
        x = tt_lib.tensor.add(x, tmp)

        return x
