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
        random_drop=0.1,
        fp_ratio=0.3,
        update_query_pos=True,
    ):
        super(QueryInteractionModule, self).__init__()
        self._build_layers(dim_in, hidden_dim, dim_out)
        self.random_drop = random_drop
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
            # ffn: linear_pos2
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
        # track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        # update ref_pts using track_instances.pred_boxes
        return track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Inescalationstances = data["init_track_instances"]
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances


# class QueryInteractionModule(nn.Module):
#     # def __init__(self, args, dim_in, hidden_dim, dim_out):
#     def __init__(self,
#     dim_in=256,
#     hidden_dim=256,
#     dim_out=256,
#     random_drop=0.1,
#     fp_ratio=0.3,
#     update_query_pos=True,
#     merger_dropout = 0,
# ):
#         super().__init__()
#         self.random_drop = random_drop
#         self.fp_ratio = fp_ratio
#         self.update_query_pos = update_query_pos
#         dropout = merger_dropout

#         self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
#         self.linear1 = nn.Linear(dim_in, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, dim_in)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(dim_in)
#         self.norm2 = nn.LayerNorm(dim_in)

#         self.activation = F.relu

#         if self.update_query_pos:
#             self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
#             self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
#             self.dropout_pos1 = nn.Dropout(dropout)
#             self.dropout_pos2 = nn.Dropout(dropout)
#             self.norm_pos = nn.LayerNorm(dim_in)

#         self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
#         self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
#         self.dropout_feat1 = nn.Dropout(dropout)
#         self.dropout_feat2 = nn.Dropout(dropout)
#         self.norm_feat = nn.LayerNorm(dim_in)

#     def _select_active_tracks(self, obj_idxes: torch.Tensor, ious: torch.Tensor, training: bool):
#         if training:
#             mask = (obj_idxes >= 0) & (ious > 0.5)
#         else:
#             mask = obj_idxes >= 0
#         return mask

#     def _update_track_embedding(self, query: torch.Tensor, output_embedding: torch.Tensor) -> torch.Tensor:
#         if query.shape[0] == 0:
#             return query

#         dim = query.shape[1]
#         query_pos = query[:, :dim // 2]
#         query_feat = query[:, dim // 2:]

#         q = k = query_pos + output_embedding
#         tgt = output_embedding
#         tgt2, _ = self.self_attn(q.unsqueeze(1), k.unsqueeze(1), tgt.unsqueeze(1))
#         tgt2 = tgt2.squeeze(1)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)

#         if self.update_query_pos:
#             query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
#             query_pos = query_pos + self.dropout_pos2(query_pos2)
#             query_pos = self.norm_pos(query_pos)

#         query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
#         query_feat = query_feat + self.dropout_feat2(query_feat2)
#         query_feat = self.norm_feat(query_feat)

#         updated_query = torch.cat([query_pos, query_feat], dim=1)
#         return updated_query

#     def forward(
#         self,
#         query: torch.Tensor,
#         output_embedding: torch.Tensor,
#         obj_idxes: torch.Tensor,
#         ious: torch.Tensor,
#         training: bool = False,
#     ) -> torch.Tensor:
#         """
#         Args:
#             query: Tensor of shape [N, C]
#             output_embedding: Tensor of shape [N, C]
#             obj_idxes: Tensor of shape [N]
#             ious: Tensor of shape [N]
#         Returns:
#             updated_query: Tensor of shape [N, C] after updating valid ones
#         """
#         mask = self._select_active_tracks(obj_idxes, ious, training)

#         if mask.sum() == 0:
#             return query  # no active tracks

#         # update only active queries
#         updated_query = self._update_track_embedding(query[mask], output_embedding[mask])

#         # clone and update the active indices only
#         final_query = query.clone()
#         final_query[mask] = updated_query
#         return final_query
