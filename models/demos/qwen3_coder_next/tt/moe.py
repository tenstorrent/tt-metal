# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Mixture-of-Experts (MoE) layer for Qwen3-Coder-Next.

512 experts × top-10 routing + shared expert (feed-forward).
Each expert is a gated MLP: [gate_proj, up_proj] → SiLU → down_proj.
Shared expert is always active and added to the routed expert output.

Expert layout (single P150a, no EP):
  - All 512 experts reside on the single device.
  - Top-k gate selects active experts per token.
  - Weighted sum of expert outputs (unsorted sparse top-k).

Weight keys (per layer):
  model.layers.{i}.mlp.experts.{e}.{gate|up|down}_proj.weight
  model.layers.{i}.mlp.shared_expert.{gate|up|down}_proj.weight
  model.layers.{i}.mlp.gate.weight  → gating scalar per expert
"""

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMoE(LightweightModule):
    """
    MoE with 512 experts + 1 shared expert.

    forward(x) → shared_mlp(x) + Σ w_e * expert_e(x) for e in top-k
    """

    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16, weights_dtype=None):
        super().__init__()
        self.device = device
        if weights_dtype is None:
            weights_dtype = getattr(config, "weights_dtype", ttnn.bfloat8_b)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.shared_intermediate_size = config.shared_expert_intermediate_size

        prefix = f"model.layers.{layer_idx}.mlp"

        # ── Shared expert (always active) ──
        self._init_shared_expert(state_dict, prefix, device, weights_dtype)

        # ── Gating network ──
        gate_w_key = f"{prefix}.gate.weight"
        if gate_w_key in state_dict:
            g = state_dict[gate_w_key].float().T.contiguous()  # [hidden, num_experts]
            self.gate_w = ttnn.from_torch(
                g.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            )
            self.gate_w_cpu = state_dict[gate_w_key].float().T.contiguous()  # [hidden, num_experts]
        else:
            self.gate_w = None
            self.gate_w_cpu = None

        # ── Expert weights (all 512) ──
        self.expert_gate_w = []
        self.expert_up_w = []
        self.expert_down_w = []
        self.expert_gate_w_cpu = []
        self.expert_up_w_cpu = []
        self.expert_down_w_cpu = []

        for e in range(self.num_experts):
            e_prefix = f"{prefix}.experts.{e}"
            gate_k = f"{e_prefix}.gate_proj.weight"
            up_k = f"{e_prefix}.up_proj.weight"
            down_k = f"{e_prefix}.down_proj.weight"

            if gate_k in state_dict:
                gw = state_dict[gate_k].T.contiguous()
                uw = state_dict[up_k].T.contiguous()
                dw = state_dict[down_k].T.contiguous()

                self.expert_gate_w.append(
                    ttnn.from_torch(
                        gw.unsqueeze(0).unsqueeze(0),
                        dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
                    )
                )
                self.expert_up_w.append(
                    ttnn.from_torch(
                        uw.unsqueeze(0).unsqueeze(0),
                        dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
                    )
                )
                self.expert_down_w.append(
                    ttnn.from_torch(
                        dw.unsqueeze(0).unsqueeze(0),
                        dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
                    )
                )
                self.expert_gate_w_cpu.append(gw.float().numpy())
                self.expert_up_w_cpu.append(uw.float().numpy())
                self.expert_down_w_cpu.append(dw.float().numpy())

    def _init_shared_expert(self, state_dict, prefix, device, dtype):
        s_prefix = f"{prefix}.shared_expert"
        sg_w = state_dict[f"{s_prefix}.gate_proj.weight"].T.contiguous()
        su_w = state_dict[f"{s_prefix}.up_proj.weight"].T.contiguous()
        sd_w = state_dict[f"{s_prefix}.down_proj.weight"].T.contiguous()

        self.shared_gate_w = ttnn.from_torch(
            sg_w.unsqueeze(0).unsqueeze(0),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.shared_up_w = ttnn.from_torch(
            su_w.unsqueeze(0).unsqueeze(0),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.shared_down_w = ttnn.from_torch(
            sd_w.unsqueeze(0).unsqueeze(0),
            dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )

    def _shared_expert_forward(self, x):
        """Shared expert (always active): gate ⊗ up → down."""
        gate = ttnn.linear(x, self.shared_gate_w)
        gate = ttnn.silu(gate)
        up = ttnn.linear(x, self.shared_up_w)
        hidden = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.shared_down_w)
        ttnn.deallocate(hidden)
        return out

    def _experts_forward_cpu(self, x_cpu, topk_indices):
        """
        CPU-side expert forwarding for single-token decode.

        Args:
            x_cpu: [1, 1, 1, hidden] flattened → [hidden]
            topk_indices: list of (expert_idx, weight) tuples

        Returns:
            [1, 1, 1, hidden] as torch tensor, bfloat16
        """
        x = x_cpu.flatten().float()  # [hidden]
        out = torch.zeros(self.hidden_size, dtype=torch.float32)

        for e_idx, weight in topk_indices:
            g = torch.sigmoid(x @ torch.tensor(self.expert_gate_w_cpu[e_idx]).T)
            u = x @ torch.tensor(self.expert_up_w_cpu[e_idx]).T
            h = g * u
            o = h @ torch.tensor(self.expert_down_w_cpu[e_idx]).T
            out += weight * o

        return out.reshape(1, 1, 1, -1).to(torch.bfloat16)

    def _topk_gate_cpu(self, x_cpu):
        """Compute top-k gating weights (CPU). Returns list of (expert_idx, weight)."""
        x = x_cpu.flatten().float()
        logits = x @ self.gate_w_cpu.T  # [num_experts]
        probs = torch.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(probs, self.num_experts_per_tok)
        return list(zip(topk_indices.tolist(), topk_values.tolist()))

    def forward(self, x):
        """
        MoE forward (decode path).
        1. Shared expert (on device).
        2. Top-k gate → select experts → weighted sum (CPU fallback).
        3. Add shared expert output + routed expert output.
        """
        # Shared expert (device)
        shared_out = self._shared_expert_forward(x)

        # Gate + routed experts (CPU fallback for now — top-k select + expert MLPs)
        x_cpu = ttnn.to_torch(x)
        topk = self._topk_gate_cpu(x_cpu)
        expert_out_cpu = self._experts_forward_cpu(x_cpu, topk)

        expert_out = ttnn.from_torch(
            expert_out_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

        # Residual add: shared + routed
        output = ttnn.add(shared_out, expert_out)
        ttnn.deallocate(shared_out)
        ttnn.deallocate(expert_out)
        return output
