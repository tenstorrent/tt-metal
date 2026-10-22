# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 MoE layer.
# Extracted from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     - HunyuanMoE   lines 1142-1232  (eager "DeepSeekMoE" path)
#
# Golden reference for the TT-Metal MoE layer (gate -> scatter -> experts ->
# weighted gather -> + shared MLP). The flashinfer / cuda-fused branch is
# omitted; we replicate only the eager path that runs on CPU/GPU verbatim.
#
# forward():
#     shared = shared_mlp(x)                          # if use_mixed_mlp_moe
#     topk_w, topk_idx = gate(x, topk_impl='easy')    # [tokens, k]
#     x_rep = x.view(-1, H).repeat_interleave(k, dim=0)   # [tokens*k, H]
#     for e in experts: out[idx==e] = experts[e](x_rep[idx==e])
#     combined = (out.view(tokens, k, H) * topk_w[..., None]).sum(1)
#     return shared + combined

import torch
import torch.nn as nn

try:
    from .gate import HunyuanTopKGate
    from .mlp import HunyuanMLP
except ImportError:  # allow running this file directly as a script
    from gate import HunyuanTopKGate
    from mlp import HunyuanMLP


class HunyuanMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        moe_topk: int,
        num_shared_expert: int = 1,
        use_mixed_mlp_moe: bool = True,
        hidden_act: str = "silu",
        mlp_bias: bool = False,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_topk = moe_topk
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe

        if use_mixed_mlp_moe:
            self.shared_mlp = HunyuanMLP(
                hidden_size,
                moe_intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                num_shared_expert=num_shared_expert,
                is_shared_mlp=True,
            )
        self.gate = HunyuanTopKGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            moe_topk=moe_topk,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                HunyuanMLP(
                    hidden_size,
                    moe_intermediate_size,
                    hidden_act=hidden_act,
                    mlp_bias=mlp_bias,
                    is_moe=True,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        input_hidden_states = hidden_states

        if self.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        # DeepSeekMoE eager implementation.
        with torch.no_grad():
            topk_weights, topk_idx = self.gate(hidden_states, topk_impl="easy")
        topk_weights = topk_weights.to(hidden_states.dtype)

        flat_topk_idx = topk_idx.view(-1)
        hidden_states_flat = input_hidden_states.view(-1, hidden_size)  # [tokens, H]
        hidden_states_repeated = hidden_states_flat.repeat_interleave(self.moe_topk, dim=0)  # [tokens*k, H]

        expert_outputs = torch.zeros_like(hidden_states_repeated)
        for i in range(self.num_experts):
            expert_mask = flat_topk_idx == i
            selected_inputs = hidden_states_repeated[expert_mask]
            expert_output = self.experts[i](selected_inputs)  # ok on zero-row tensor
            expert_outputs[expert_mask] = expert_output

        combined_output = (
            expert_outputs.view(bsz * seq_len, self.moe_topk, hidden_size) * topk_weights.unsqueeze(-1)
        ).sum(dim=1)
        combined_output = combined_output.to(hidden_states.dtype).view(bsz, seq_len, hidden_size)

        if self.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, H, I = 1, 32, 4096, 3072
    NE, K = 8, 2  # small for a fast smoke test
    x = torch.randn(B, S, H, dtype=torch.bfloat16)

    moe = HunyuanMoE(H, I, num_experts=NE, moe_topk=K).to(torch.bfloat16).eval()
    with torch.no_grad():
        out = moe(x)
    print(f"input  : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"experts: {NE}  topk: {K}  shared_mlp: {moe.use_mixed_mlp_moe}")
    print(f"output : {tuple(out.shape)}  dtype={out.dtype}")
