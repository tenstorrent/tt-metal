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
#
# Host RAM: streaming still needs expert tensors to rebuild MLPs each forward.
# Retaining the full layer state_dict across a 32-layer stack pins ~150–200GB and
# gets SIGTERM'd by the OOM killer mid-backbone-load. Call ``bind_expert_loader``
# (done by HunyuanTtModel) to swap retained tensors for on-demand disk reads.

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
        weight_cache_path=None,
    ):
        super().__init__()
        # Accepted for a uniform constructor signature with
        # HunyuanTtMoEParallel (transformer_layer.py passes it to both). The
        # single-device path loads experts directly from state_dict, so there
        # is no disk cache to key.
        self.weight_cache_path = weight_cache_path
        self.device = device
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.weight_dtype = weight_dtype
        self.stream_experts = stream_experts
        self.prefix = prefix
        self._expert_loader = None

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
            # Experts now live on device; drop the host-RAM reference to this
            # layer's torch weights so the backbone loader's `del sd; gc.collect()`
            # can actually free them.
            self.state_dict = None
        else:
            # Keep expert tensors only (gate/shared already uploaded). Single-layer
            # PCC fits; multi-layer stacks must call bind_expert_loader().
            expert_pfx = f"{prefix}.experts."
            self.state_dict = {k: v for k, v in state_dict.items() if k.startswith(expert_pfx)}

    def bind_expert_loader(self, loader):
        """Use ``loader(expert_idx) -> state_dict`` and drop retained host experts.

        ``loader`` must return a dict containing
        ``{prefix}.experts.{e}.gate_and_up_proj.weight`` and
        ``{prefix}.experts.{e}.down_proj.weight`` for the requested expert.
        """
        if not self.stream_experts:
            raise RuntimeError("bind_expert_loader requires stream_experts=True")
        self._expert_loader = loader
        self.state_dict = None

    def _expert_state_dict(self, expert_idx: int) -> dict:
        if self._expert_loader is not None:
            return self._expert_loader(expert_idx)
        if self.state_dict is None:
            raise RuntimeError("streaming MoE has no state_dict or expert_loader")
        return self.state_dict

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
                    self._expert_state_dict(e),
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
