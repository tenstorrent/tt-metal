# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn
import torch.nn.functional as F
from models.experimental.uniad.reference.utils import Instances


class QueryInteractionModule(nn.Module):
    def __init__(
        self,
        dim_in=256,
        hidden_dim=256,
        dim_out=256,
        fp_ratio=0.3,
        update_query_pos=True,
    ):
        super(QueryInteractionModule, self).__init__()
        self._build_layers(dim_in, hidden_dim, dim_out)
        self.fp_ratio = fp_ratio
        self.update_query_pos = update_query_pos

    def _build_layers(self, dim_in, hidden_dim, dim_out, update_query_pos=True):
        self.self_attn = nn.MultiheadAttention(dim_in, 8)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)

            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)

        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, : dim // 2]
        query_feat = track_instances.query[:, dim // 2 :]
        q = k = query_pos + out_embed

        # attention
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]

        tgt = self.norm1(tgt)

        # ffn
        x = self.linear1(tgt)
        x = self.activation(x)
        tgt2 = self.linear2(x)

        tgt = self.norm2(tgt)

        if self.update_query_pos:
            x = self.linear_pos1(tgt)
            x = self.activation(x)
            query_pos2 = self.linear_pos2(x)
            query_pos = self.norm_pos(query_pos)
            track_instances.query[:, : dim // 2] = query_pos

        x = self.linear_feat1(tgt)
        x = self.activation(x)
        query_feat2 = self.linear_feat2(x)
        query_feat = self.norm_feat(query_feat)
        track_instances.query[:, dim // 2 :] = query_feat
        return track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances
