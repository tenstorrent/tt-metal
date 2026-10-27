# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MAF_Block(nn.Module):
    def __init__(
        self, dim: int, num_experts_per_modality: int, num_heads: int, num_fusion_layers: int, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3

        self.gating_network = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid())

        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)

        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.MultiheadAttention) and hasattr(m, "out_proj"):
            torch.nn.init.zeros_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                torch.nn.init.zeros_(m.out_proj.bias)

    def forward(
        self, video_tokens: torch.Tensor, text_tokens: torch.Tensor, audio_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = video_tokens.shape[0]

        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)

        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)

        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        updated_experts = self.norm1(experts + info)

        fused_experts = self.fusion_transformer(updated_experts)

        fused_v_experts, fused_t_experts, fused_a_experts = fused_experts.chunk(3, dim=1)

        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        alpha_v = torch.sigmoid(self.bypass_gate_v)
        alpha_t = torch.sigmoid(self.bypass_gate_t)
        alpha_a = torch.sigmoid(self.bypass_gate_a)

        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a,
        }
