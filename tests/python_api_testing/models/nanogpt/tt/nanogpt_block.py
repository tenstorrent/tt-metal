import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
import python_api_testing.models.nanogpt.tt.nanogpt_attention as nanogpt_attention

from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass
import math

from transformers import GPT2LMHeadModel

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class TtBlock(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.device = device

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

        self.mlp = nanogpt_mlp.TtMLP(f"{base_address}.mlp", state_dict, device)

    def forward(self, x):
        tmp = self.attn.forward(self.ln_1(x))
        x = tt_lib.tensor.add(x, tmp)

        tmp = self.mlp.forward(self.ln_2(x))
        x = tt_lib.tensor.add(x, tmp)

        return x
