# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 MoE layer.
# Mirrors ref/moe/moe.py (eager DeepSeekMoE path):
#     shared      = shared_mlp(x)                       # if use_mixed_mlp_moe
#     w, idx      = gate(x)                              # top-k routing
#     combined    = sum_e expert_e(x) * combine_w[:, e]  # combine_w[t,e]=routed
#                                                        #   weight if e in top-k
#                                                        #   else 0
#     out         = shared + combined
#
# Numerical equivalence to the reference gather/scatter:
#   The reference runs expert e only on the tokens routed to it and weights
#   them by the (normalised) routing prob. Here we run expert e on ALL tokens
#   and multiply by combine_w[:, e], which is exactly 0 for tokens that did not
#   select e. So the unselected contributions are identically zero — the result
#   matches the gather/scatter implementation up to matmul precision.
#
# This is the correctness-reference port (dense over experts). `stream_experts`
# loads each expert's weights, runs it, and frees them before the next expert,
# bounding device memory to ~one expert at a time (needed for the 64-expert
# real-weight layer); set it False to pre-load all experts for speed.

import ttnn
from models.common.lightweightmodule import LightweightModule

from .gate import HunyuanTtTopKGate
from .mlp import HunyuanTtMLP


class HunyuanTtMoE(LightweightModule):
    def __init__(
        self,
        device,
        hidden_size: int,
        num_experts: int,
        moe_topk: int,
        state_dict: dict,
        prefix: str,
        use_mixed_mlp_moe: bool = True,
        norm_topk_prob: bool = True,
        weight_dtype=ttnn.bfloat16,
        stream_experts: bool = True,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.weight_dtype = weight_dtype
        self.stream_experts = stream_experts
        self.state_dict = state_dict
        self.prefix = prefix

        self.gate = HunyuanTtTopKGate(
            device,
            hidden_size,
            num_experts,
            moe_topk,
            state_dict,
            f"{prefix}.gate.wg",
            norm_topk_prob=norm_topk_prob,
        )

        if use_mixed_mlp_moe:
            self.shared_mlp = HunyuanTtMLP(
                device, hidden_size, state_dict, f"{prefix}.shared_mlp", weight_dtype=weight_dtype
            )

        # Pre-load experts only when not streaming.
        self.experts = None
        if not stream_experts:
            self.experts = [
                HunyuanTtMLP(device, hidden_size, state_dict, f"{prefix}.experts.{i}", weight_dtype=weight_dtype)
                for i in range(num_experts)
            ]

    def _gate_weights(self, x):
        """Run the gate once and return its top-k routing on device:
            topk_w  : [B, S, k] bf16  normalised routed weights
            topk_idx: [B, S, k] bf16  selected expert ids (cast for comparison)
        The per-expert combine weight is later derived on device as
            w_e[t] = sum_k topk_w[t,k] * (topk_idx[t,k] == e).
        """
        topk_w_t, topk_idx_t = self.gate(x)
        topk_idx_f = ttnn.typecast(topk_idx_t, ttnn.bfloat16)  # ids <= 63 are exact in bf16
        ttnn.deallocate(topk_idx_t)
        return topk_w_t, topk_idx_f

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: TTNN tensor [B, S, H] in TILE_LAYOUT.
        Returns:
            [B, S, H] tensor.
        """
        topk_w, topk_idx = self._gate_weights(x)  # [B, S, k] each, on device

        combined_out = None
        for e in range(self.num_experts):
            # Per-token combine weight for expert e: select the matching top-k
            # slots and sum them. Tokens that did not pick e get 0 (exact).
            sel = ttnn.eq(topk_idx, float(e))  # [B, S, k]
            contrib = ttnn.multiply(sel, topk_w)  # [B, S, k]
            ttnn.deallocate(sel)
            w_e = ttnn.sum(contrib, dim=-1, keepdim=True)  # [B, S, 1]
            ttnn.deallocate(contrib)

            expert = (
                self.experts[e]
                if self.experts is not None
                else HunyuanTtMLP(
                    self.device,
                    self.hidden_size,
                    self.state_dict,
                    f"{self.prefix}.experts.{e}",
                    weight_dtype=self.weight_dtype,
                )
            )
            oe = expert(x)  # [B, S, H]

            weighted = ttnn.multiply(oe, w_e)  # broadcast [B, S, 1] over H
            ttnn.deallocate(oe)
            ttnn.deallocate(w_e)

            if combined_out is None:
                combined_out = weighted
            else:
                tmp = ttnn.add(combined_out, weighted)
                ttnn.deallocate(combined_out)
                ttnn.deallocate(weighted)
                combined_out = tmp

            if self.experts is None:
                expert.deallocate()

        ttnn.deallocate(topk_w)
        ttnn.deallocate(topk_idx)

        if self.use_mixed_mlp_moe:
            shared = self.shared_mlp(x)
            out = ttnn.add(shared, combined_out)
            ttnn.deallocate(shared)
            ttnn.deallocate(combined_out)
            return out

        return combined_out
