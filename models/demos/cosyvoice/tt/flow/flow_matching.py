# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flow model wrapper — full tokens→mu pipeline (Stage 1, host-side).

Combines:
  - input_embedding: Embedding(6561, 512)
  - spk_embed_affine_layer: Linear(192, 80)
  - encoder: UpsampleConformerEncoder
  - encoder_proj: Linear(512, 80)

Produces `mu` [1, 80, T_mel] from speech tokens + speaker embedding.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from models.demos.cosyvoice.tt.flow.encoder import UpsampleConformerEncoder
from models.demos.cosyvoice.tt.model_config import FLOW


class FlowEncoderModel(torch.nn.Module):
    """Full flow encoder pipeline: tokens + spk → mu."""

    def __init__(self, flow_weights: Dict[str, Dict[str, torch.Tensor]]):
        super().__init__()
        ie_w = flow_weights["input_embedding"]["input_embedding.weight"]
        self.input_embedding = torch.nn.Embedding(ie_w.shape[0], ie_w.shape[1])
        self.input_embedding.weight = torch.nn.Parameter(ie_w)

        sa_w = flow_weights["spk_embed_affine_layer"]["spk_embed_affine_layer.weight"]
        sa_b = flow_weights["spk_embed_affine_layer"]["spk_embed_affine_layer.bias"]
        self.spk_embed_affine_layer = torch.nn.Linear(sa_w.shape[1], sa_w.shape[0])
        self.spk_embed_affine_layer.weight = torch.nn.Parameter(sa_w)
        self.spk_embed_affine_layer.bias = torch.nn.Parameter(sa_b)

        self.encoder = UpsampleConformerEncoder(
            flow_weights["encoder"],
            n_heads=FLOW.encoder.attention_heads,
            d_model=FLOW.encoder.output_size,
            ffn_dim=FLOW.encoder.linear_units,
            n_blocks=FLOW.encoder.num_blocks,
            n_up_blocks=4,
        )

        ep_w = flow_weights["encoder_proj"]["encoder_proj.weight"]
        ep_b = flow_weights["encoder_proj"]["encoder_proj.bias"]
        self.encoder_proj = torch.nn.Linear(ep_w.shape[1], ep_w.shape[0])
        self.encoder_proj.weight = torch.nn.Parameter(ep_w)
        self.encoder_proj.bias = torch.nn.Parameter(ep_b)

    PRE_LOOKAHEAD_LEN = 3

    @torch.no_grad()
    def forward(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        streaming: bool = False,
        finalize: bool = True,
    ) -> torch.Tensor:
        """Flow encoder forward.

        Args:
            token: [1, T_gen] generated speech tokens
            token_len: [1] length
            prompt_token: [1, T_prompt] prompt speech tokens
            prompt_token_len: [1] length
            prompt_feat: [1, T_prompt_mel, 80] prompt mel
            prompt_feat_len: [1] length
            embedding: [1, 192] speaker embedding
            streaming: use chunked-causal attention in encoder
            finalize: if False, split last PRE_LOOKAHEAD_LEN tokens as context

        Returns:
            mu: [1, 80, T_mel] — encoder output (transposed for estimator)
            spks: [1, 80] — speaker embedding
            conds: [1, 80, T_mel] — prompt condition
        """
        embedding = F.normalize(embedding, dim=1)
        spks = self.spk_embed_affine_layer(embedding)

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~self._make_pad_mask(token_len, token.shape[1])).unsqueeze(-1).to(spks.dtype)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        context = None
        if streaming and not finalize:
            context = token[:, -self.PRE_LOOKAHEAD_LEN :]
            token = token[:, : -self.PRE_LOOKAHEAD_LEN]

        h = self.encoder(token, streaming=streaming, context=context)
        h = self.encoder_proj(h)

        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        conds = torch.zeros([1, mel_len1 + mel_len2, 80], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mu = h.transpose(1, 2).contiguous()
        return mu, spks, conds

    @staticmethod
    def _make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create padding mask. True = padded position."""
        arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return arange >= lengths.unsqueeze(1)
