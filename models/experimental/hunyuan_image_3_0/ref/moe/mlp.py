# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 MLP (expert FFN + shared MLP).
# Extracted from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     - HunyuanMLP   lines 1046-1090
#
# Used as the golden reference for the TT-Metal expert FFN port.
#
# Structure (hidden_act="silu" -> SwiGLU):
#     gate_and_up_proj: Linear(hidden, 2 * I, bias=mlp_bias)
#     x1, x2 = gate_and_up_proj(x).chunk(2, dim=-1)
#     down_proj:        Linear(I, hidden, bias=mlp_bias)
#     out = down_proj(x1 * silu(x2))
# where:
#     I = moe_intermediate_size                    (routed expert, is_moe=True)
#     I = moe_intermediate_size * num_shared_expert (shared MLP, is_shared_mlp=True)
#
# Layer-0 of the shipped checkpoint: hidden=4096, moe_intermediate_size=3072,
# num_shared_expert=1, mlp_bias=False, so:
#     expert.gate_and_up_proj.weight : [6144, 4096]   (2*3072)
#     expert.down_proj.weight        : [3072, 4096]
#     shared_mlp identical shapes (num_shared_expert=1).

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class HunyuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        mlp_bias: bool = False,
        num_shared_expert: int = 1,
        is_shared_mlp: bool = False,
        is_moe: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act

        self.intermediate_size = intermediate_size
        if is_shared_mlp or is_moe:
            # moe / shared paths use moe_intermediate_size (passed in here).
            if is_shared_mlp:
                self.intermediate_size *= num_shared_expert

        self.act_fn = ACT2FN[hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=mlp_bias)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        else:
            assert False, "other hidden_act are not supported"

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=-1)
            down_proj = self.down_proj(x1 * self.act_fn(x2))
            return down_proj
        elif self.hidden_act == "gelu":
            intermediate = self.gate_and_up_proj(x)
            intermediate = self.act_fn(intermediate)
            output = self.down_proj(intermediate)
            return output
        else:
            assert False, "other hidden_act are not supported"


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, H, I = 1, 256, 4096, 3072
    x = torch.randn(B, S, H, dtype=torch.bfloat16)

    expert = HunyuanMLP(H, I, is_moe=True).to(torch.bfloat16).eval()
    with torch.no_grad():
        out = expert(x)
    print(f"input              : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"gate_and_up_proj.w : {tuple(expert.gate_and_up_proj.weight.shape)}")
    print(f"down_proj.w        : {tuple(expert.down_proj.weight.shape)}")
    print(f"output             : {tuple(out.shape)}  dtype={out.dtype}")
