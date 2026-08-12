# SPDX-FileCopyrightText: © 2025-2026 The Moonshot AI Team, DeepSeek-AI, and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The mixture-of-experts block in this file is adapted from DeepSeek-V3
# (DeepSeek-V3/modeling_deepseek.py). It has been modified and extended for the Kimi-Linear
# architecture, notably with the shared low-rank latent projection pair ("LatentMoE").
#
# Licensing Information:
# - Code adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the Kimi K3 License (see the LICENSE file in this repository).
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi-K3 MoE reference, trimmed from upstream ``modeling_kimi_linear.py``.

Provenance: ``huggingface.co/moonshotai/Kimi-K3``, ``modeling_kimi_linear.py`` -- ``SituAndMul``
(upstream lines 64-82), ``_get_situ_activation_params`` (88-91), ``KimiBlockSparseMLP`` (242-270),
``KimiMLP`` (273-301), ``KimiMoEGate`` (666-761) and ``KimiSparseMoeBlock`` (762-873). The MoE math
is unchanged; this is the truth model the TT latent-MoE path is compared against.

Why trimmed rather than vendored whole: upstream raises
``ImportError("Plese run `pip install -U fla-core`")`` at *module import* if ``fla`` is absent -- a
triton/GPU linear-attention library, not installed here and needed only by ``KimiDeltaAttention``
(the KDA layers, out of scope). See the sibling ``modeling_kimi_k3_mla.py`` for the same reasoning.

``KimiRMSNorm`` is imported from that sibling rather than copied, so the two references cannot drift.

One deliberate deviation from upstream: upstream registers the activation globally with
``ACT2FN["situ"] = SituAndMul`` at module scope. That mutates a dict shared with the rest of
``transformers`` as an import side effect, so it is omitted here. It is dead code for this file
anyway -- both MLP classes construct ``SituAndMul`` directly when ``hidden_act == "situ"`` and only
consult ``ACT2FN`` on the non-situ (e.g. plain SiLU) path.

What "LatentMoE" is, since it is the whole point of the K3 delta: ``routed_expert_hidden_size``
(3584) being set makes the **routed** experts run in a reduced latent space -- ``down_proj`` 7168 ->
3584 before dispatch, the top-k weighted sum and ``routed_expert_norm`` in latent space, then one
shared ``up_proj`` 3584 -> 7168. The router still reads the full 7168 hidden, and the shared expert
is untouched: a dense MLP at 7168 over the *pre*-projection input.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiRMSNorm


class SituAndMul(nn.Module):
    """
    SituAndMul activation: beta * tanh(gate / beta) * sigmoid(gate) * up
    When linear_beta is set, up is also transformed by linear_beta * tanh(up / linear_beta).
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


def _get_situ_activation_params(config: KimiLinearConfig):
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return beta or 1.0, linear_beta


class KimiBlockSparseMLP(nn.Module):
    """One routed expert. Note the K3 weight names: w1 = gate, w3 = up, w2 = down."""

    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # up

        if config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
            current_hidden_states = self.act_fn(gate_up)
        else:
            current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class KimiMLP(nn.Module):
    """Dense MLP -- used for the shared expert (and for layer 0's dense FFN upstream)."""

    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            down_proj = self.down_proj(self.act_fn(gate_up))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim)),
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}",
            )

        # select top-k experts
        assert not self.training
        scores = scores.view(bsz * seq_len, -1)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, num_expert_group]
            group_idx = torch.topk(
                group_scores,
                k=self.topk_group,
                dim=-1,
                sorted=False,
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
        else:
            tmp_scores = scores_for_choice
        _, topk_idx = torch.topk(
            tmp_scores,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V3's MOE implementation
    The namings are consistent with Kimi's version.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        self.ep_size = 1
        self.experts_per_rank = config.num_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ],
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config,
                intermediate_size=intermediate_size,
            )

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(
                    self.moe_hidden_size,
                    eps=config.rms_norm_eps,
                )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        else:
            raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
