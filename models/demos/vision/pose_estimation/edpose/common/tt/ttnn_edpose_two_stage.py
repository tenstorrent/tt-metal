# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage query generation for ED-Pose (CPU).

After the encoder processes multi-scale features, two-stage selection picks
the top-K (900) proposals from encoder output to initialize decoder queries.
This includes:
  1. enc_output + enc_output_norm projections
  2. Class scoring per token to select top-K
  3. Box proposal generation (reference points)
  4. Target (tgt) embedding initialization

All operations run on CPU as they involve dynamic indexing (topk, gather).
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

EDPOSE_ROOT = os.environ.get("EDPOSE_ROOT", os.path.expanduser("~/ttwork/ED-Pose"))


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, learnedwh=None):
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory.clone()
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
    output_memory = output_memory.masked_fill(~output_proposals_valid, 0.0)
    return output_memory, output_proposals


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TwoStageQueryGenerator:
    """
    Generates decoder queries from encoder output using two-stage standard approach.

    Loads enc_output, enc_output_norm, enc_out_class_embed, enc_out_bbox_embed,
    and tgt_embed from the ED-Pose checkpoint.

    When spatial_shapes are known at init time and mask is all-False (no padding),
    proposals are pre-computed once.
    """

    def __init__(self, state_dict, d_model=256, num_queries=900, num_classes=2,
                 spatial_shapes=None):
        self.d_model = d_model
        self.num_queries = num_queries

        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_out_class_embed = nn.Linear(d_model, num_classes)
        self.enc_out_bbox_embed = MLP(d_model, d_model, 4, 3)

        self.enc_output.load_state_dict({
            "weight": state_dict["transformer.enc_output.weight"],
            "bias": state_dict["transformer.enc_output.bias"],
        })
        self.enc_output_norm.load_state_dict({
            "weight": state_dict["transformer.enc_output_norm.weight"],
            "bias": state_dict["transformer.enc_output_norm.bias"],
        })

        class_sd = {k.replace("transformer.enc_out_class_embed.", ""): v
                     for k, v in state_dict.items() if k.startswith("transformer.enc_out_class_embed.")}
        self.enc_out_class_embed.load_state_dict(class_sd)

        bbox_sd = {k.replace("transformer.enc_out_bbox_embed.", ""): v
                    for k, v in state_dict.items() if k.startswith("transformer.enc_out_bbox_embed.")}
        self.enc_out_bbox_embed.load_state_dict(bbox_sd)

        self.tgt_embed_weight = state_dict.get("transformer.tgt_embed.weight", None)

        self.enc_output.eval()
        self.enc_output_norm.eval()
        self.enc_out_class_embed.eval()
        self.enc_out_bbox_embed.eval()

        self._cached_proposals = None
        self._cached_valid_mask = None
        if spatial_shapes is not None:
            self._cache_proposals(spatial_shapes)

    def _cache_proposals(self, spatial_shapes):
        proposals = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            H_, W_ = int(H_), int(W_)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)
            grid = (grid + 0.5) / torch.tensor([W_, H_], dtype=torch.float32)
            wh = torch.full_like(grid, 0.05 * (2.0 ** lvl))
            proposal = torch.cat((grid, wh), -1).view(1, -1, 4)
            proposals.append(proposal)
        output_proposals = torch.cat(proposals, 1)
        valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(~valid, float("inf"))
        self._cached_proposals = output_proposals
        self._cached_valid_mask = valid

    @torch.no_grad()
    def __call__(self, memory, mask_flatten, spatial_shapes, embed_init_tgt=False):
        """
        Generate decoder queries from encoder memory.

        Args:
            memory: (N, sum(Hi*Wi), d_model) — encoder output
            mask_flatten: (N, sum(Hi*Wi)) — bool padding mask
            spatial_shapes: (num_levels, 2)

        Returns dict:
            tgt: (N, num_queries, d_model) — decoder target embeddings
            refpoint_embed: (N, num_queries, 4) — unsigmoided reference points
            init_box_proposal: (N, num_queries, 4) — sigmoided box proposals
            hs_enc: (1, N, num_queries, d_model)
            ref_enc: (1, N, num_queries, 4)
        """
        no_padding = not mask_flatten.any().item()

        if self._cached_proposals is None:
            self._cache_proposals(spatial_shapes)

        if no_padding:
            output_proposals = self._cached_proposals.expand(memory.shape[0], -1, -1)
            output_memory = memory * self._cached_valid_mask.float()
        else:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_outputs_class = self.enc_out_class_embed(output_memory)
        enc_outputs_coord = self.enc_out_bbox_embed(output_memory) + output_proposals

        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        refpoint_embed = torch.gather(
            enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).detach()
        init_box_proposal = torch.gather(
            output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid()

        tgt_undetach = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
        )

        if embed_init_tgt and self.tgt_embed_weight is not None:
            bs = memory.shape[0]
            tgt = self.tgt_embed_weight[None, :, :].repeat(bs, 1, 1)
        else:
            tgt = tgt_undetach.detach()

        hs_enc = tgt_undetach.unsqueeze(0)
        ref_enc = torch.gather(
            enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid().unsqueeze(0)

        return {
            "tgt": tgt,
            "refpoint_embed": refpoint_embed,
            "init_box_proposal": init_box_proposal,
            "hs_enc": hs_enc,
            "ref_enc": ref_enc,
        }
