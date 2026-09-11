# SPDX-FileCopyrightText: © 2025-2026 The Moonshot AI Team, DeepSeek-AI, and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The multi-head latent attention in this file is adapted from DeepSeek-V3
# (DeepSeek-V3/modeling_deepseek.py). It has been extensively modified and extended for the
# Kimi-Linear architecture.
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

"""Kimi-K3 full-attention (MLA) reference, trimmed from upstream ``modeling_kimi_linear.py``.

Provenance: ``huggingface.co/moonshotai/Kimi-K3``, ``modeling_kimi_linear.py`` -- ``KimiRMSNorm``
(upstream lines 226-236) and ``KimiMLAAttention`` (upstream lines 335-474). The MLA math is
unchanged; this is the **unabsorbed** truth model used as an independent cross-check on the
absorbed ``MLAReference`` that mirrors the device op order.

Why trimmed rather than vendored whole:
  * upstream raises ``ImportError("Plese run `pip install -U fla-core`")`` at *module import* if
    ``fla`` is absent -- it is a triton/GPU linear-attention library, not installed here, and
    needed only by ``KimiDeltaAttention`` (the KDA layers, out of scope).
  * upstream routes attention through ``ALL_ATTENTION_FUNCTIONS`` / ``transformers.masking_utils``,
    whose surface moves between transformers majors (this repo runs 5.x). Replaced with a direct
    ``F.scaled_dot_product_attention``, which is what the eager path computes anyway.

Everything else about the MLA layer is preserved verbatim, in particular the three K3 deltas:
  * ``self.scaling = self.q_head_dim ** (-0.5)`` -- NO mscale (K2.6 multiplies by ~2.0)
  * ``self.rotary_emb = None`` and ``assert self.use_nope``: ``k_rot`` is expanded across heads and
    concatenated **unrotated**; the 64 rope columns still occupy the tensor
  * the output gate multiplies in ``num_heads * v_head_dim`` space, **before** ``o_proj``
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class KimiMLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int = 0):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(f"Kimi MLA config is not found or not properly formatted: {e}")

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.is_causal = True
        assert self.use_nope

        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        if self.use_output_gate:
            projection_size = self.num_heads * self.v_head_dim
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # NoPE: k_rot is broadcast across all heads and concatenated WITHOUT rotation.
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Upstream dispatches via ALL_ATTENTION_FUNCTIONS; the eager path computes exactly this.
        # q/k are q_head_dim (192) wide while v is v_head_dim (128) wide, which SDPA supports.
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=attention_mask is None and self.is_causal,
            scale=self.scaling,
        ).transpose(1, 2)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)
        return attn_output
