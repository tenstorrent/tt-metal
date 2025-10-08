# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn

from models.experimental.uniad.reference.utils import Instances


class MemoryBank(nn.Module):
    def __init__(
        self,
        dim_in=256,
        hidden_dim=256,
        dim_out=256,
        memory_bank_score_thresh=0,
        memory_bank_len=4,
    ):
        super().__init__()
        self._build_layers(memory_bank_score_thresh, memory_bank_len, dim_in, hidden_dim, dim_out)

    def _build_layers(self, memory_bank_score_thresh, memory_bank_len, dim_in, hidden_dim, dim_out):
        self.save_thresh = memory_bank_score_thresh
        self.save_period = 3
        self.max_his_length = memory_bank_len

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def update(self, track_instances):
        embed = track_instances.output_embedding[:, None]  # ( N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat(
                [
                    mem_padding_mask[saved_idxes, 1:],
                    torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def _forward_temporal_attn(self, track_instances):
        key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]

        valid_idxes = key_padding_mask[:, -1] == 0

        valid_indices = valid_idxes.nonzero(as_tuple=False).squeeze(-1)
        embed = torch.index_select(track_instances.output_embedding, dim=0, index=valid_indices)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),  # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
                key_padding_mask=key_padding_mask,
            )[0][0]

            out = embed + embed2
            embed = self.temporal_norm1(out)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))  ##
            embed = self.temporal_norm2(embed + embed2)
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances
