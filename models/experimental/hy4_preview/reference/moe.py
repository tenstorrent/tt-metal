# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for Hy4-preview's feed-forward sublayers.

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py``: HYV4MLP,
HYV4TopkRouter, HYV4Experts and HYV4MoE.

Layer 0 is a dense SwiGLU FFN (intermediate 18432). Layers 1-77 and the MTP layer are MoE:

    scores  = sigmoid(x W_router)                           [T, 256], fp32
    picks   = top-8 of (scores + e_score_correction_bias)   the bias steers the choice only
    weights = scores[picks] / sum(scores[picks]) * 2.827    the bias is not in the weights
    routed  = sum over the 8 picks of weight * expert(x)
    out     = routed + shared_expert(x)

A routed expert is SwiGLU with a clamp: its gate is clamped from above at 10 and its up branch to
[-10, 10] before ``silu(gate) * up``. The shared expert and the dense FFN are plain SwiGLU, no
clamp. Hy4 has one expert group, so the router is a plain global top-k.

Checkpoint names under ``model.layers.{L}.mlp.``:
    dense (layer 0):  gate_proj.weight  up_proj.weight  down_proj.weight
    MoE:              gate.weight  gate.e_score_correction_bias
                      experts.gate_up_proj [E, 2I, H] (gate rows first)  experts.down_proj [E, H, I]
                      shared_experts.{gate_proj, up_proj, down_proj}.weight
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """Feed-forward hyperparameters. Defaults are the tencent/Hy4-preview config.json values."""

    hidden_size: int = 6144
    num_experts: int = 256
    top_k: int = 8
    intermediate_size: int = 2048  # one routed expert; also one shared expert
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.827
    swiglu_limit: float = 10.0
    dense_intermediate_size: int = 18432  # layer 0

    @property
    def shared_intermediate_size(self) -> int:
        return self.intermediate_size * self.n_shared_experts


class Hy4FFN(nn.Module):
    """HYV4MLP: down(silu(gate(x)) * up(x)), no bias, no clamp. The dense layer 0 and the shared expert."""

    PARAMETER_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(hidden_size, intermediate_size))
        for p in self.parameters():
            nn.init.normal_(p, std=0.006)

    def weights(self) -> dict:
        """The three weights by checkpoint name, as the device module takes them."""
        return {name: getattr(self, name.removesuffix(".weight")).detach() for name in self.PARAMETER_NAMES}

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, hidden] -> silu(gate) * up, [T, intermediate]."""
        return F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.activation(x), self.down_proj)


class Hy4MoE(nn.Module):
    """One MoE sublayer on a batch of tokens. x: [T, hidden] -> [T, hidden].

    The caller applies the layer's pre-norm (``post_attention_layernorm``) before and iHC's hc_post after.
    """

    # checkpoint name -> attribute
    PARAMETER_NAMES = {
        "gate.weight": "router_weight",
        "gate.e_score_correction_bias": "router_bias",
        "experts.gate_up_proj": "gate_up_proj",
        "experts.down_proj": "down_proj",
        "shared_experts.gate_proj.weight": "shared.gate_proj",
        "shared_experts.up_proj.weight": "shared.up_proj",
        "shared_experts.down_proj.weight": "shared.down_proj",
    }

    def __init__(self, cfg: MoEConfig, init: bool = True):
        """``init=False`` leaves the weights uninitialised, for a caller that loads all of them: at
        Hy4's size the routed experts are 9.7 billion values."""
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.router_weight = nn.Parameter(torch.empty(c.num_experts, c.hidden_size))
        self.router_bias = nn.Parameter(torch.zeros(c.num_experts))
        self.gate_up_proj = nn.Parameter(torch.empty(c.num_experts, 2 * c.intermediate_size, c.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(c.num_experts, c.hidden_size, c.intermediate_size))
        self.shared = Hy4FFN(c.hidden_size, c.shared_intermediate_size)
        if init:
            for p in (self.router_weight, self.gate_up_proj, self.down_proj):
                nn.init.normal_(p, std=0.006)

    def parameter(self, name: str) -> nn.Parameter:
        """The parameter a checkpoint name refers to."""
        return self.get_parameter(self.PARAMETER_NAMES[name])

    def weights(self) -> dict:
        """The seven parameters by checkpoint name, as the device module takes them."""
        return {name: self.parameter(name).detach() for name in self.PARAMETER_NAMES}

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, hidden] -> sigmoid router scores [T, E], fp32."""
        return F.linear(x.float(), self.router_weight.float()).sigmoid()

    def route(self, x: torch.Tensor):
        """x: [T, hidden] -> (weights [T, top_k] fp32, indices [T, top_k]).

        The correction bias is added for the choice only; the weights are the unbiased scores of the
        chosen experts, normalised to sum to one and scaled.
        """
        scores = self.scores(x)
        indices = torch.topk(scores + self.router_bias.float(), k=self.cfg.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.cfg.routed_scaling_factor, indices

    def dense_weights(self, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """(weights, indices) [T, top_k] -> [T, E]: a token's weight for every expert, zero where not picked."""
        return torch.zeros(weights.shape[0], self.cfg.num_experts, dtype=weights.dtype).scatter(1, indices, weights)

    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        """A routed expert's activation: [..., 2I] (gate | up) -> [..., I], with the clamp."""
        gate, up = gate_up.chunk(2, dim=-1)
        limit = self.cfg.swiglu_limit
        return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)

    def expert(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """One routed expert on x: [N, hidden] -> [N, hidden], unweighted."""
        return F.linear(self.swiglu(F.linear(x, self.gate_up_proj[index])), self.down_proj[index])

    def routed(self, x: torch.Tensor, dense_weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum of the picked experts. dense_weights: [T, E], zero where an expert is not picked."""
        out = torch.zeros_like(x)
        for index in dense_weights.count_nonzero(dim=0).nonzero().flatten().tolist():
            tokens = dense_weights[:, index].nonzero().flatten()
            current = self.expert(x[tokens], index) * dense_weights[tokens, index, None]
            out.index_add_(0, tokens, current.to(out.dtype))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.route(x)
        return self.routed(x, self.dense_weights(weights, indices)) + self.shared(x)
