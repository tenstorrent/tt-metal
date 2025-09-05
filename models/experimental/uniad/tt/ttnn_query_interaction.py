# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.uniad.tt.ttnn_utils import Instances
from models.experimental.uniad.tt.ttnn_multi_head_attention import TtMultiheadAttention


class TtQueryInteractionModule:
    def __init__(
        self,
        dim_in=256,
        hidden_dim=256,
        dim_out=256,
        fp_ratio=0.3,
        update_query_pos=True,
        params=None,
        device=None,
        eps=1e-05,
    ):
        self.device = device
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.fp_ratio = fp_ratio
        self.update_query_pos = update_query_pos
        self.params = params
        self.eps = eps
        self.self_attn = TtMultiheadAttention(device, params.self_attn, embed_dim=dim_in, num_heads=8)

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, : dim // 2]
        query_feat = track_instances.query[:, dim // 2 :]
        q = k = query_pos + out_embed

        # attention
        tgt = out_embed

        q = ttnn.reshape(q, (q.shape[0], 1, q.shape[1]))
        k = ttnn.reshape(k, (k.shape[0], 1, k.shape[1]))
        value = ttnn.reshape(tgt, (tgt.shape[0], 1, tgt.shape[1]))
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        if q.shape[0] > 0 and k.shape[0] > 0:
            tgt2 = self.self_attn(q, k, value=value)[0]

            tgt2 = ttnn.to_torch(tgt2)
            tgt = ttnn.to_torch(tgt)

            tgt2 = tgt2[:, 0]

            tgt = ttnn.from_torch(tgt, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            tgt2 = ttnn.from_torch(tgt2, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            tgt = ttnn.layer_norm(
                tgt,
                weight=self.params.norm1.weight,
                bias=self.params.norm1.bias,
                epsilon=self.eps,
            )

            x = ttnn.linear(tgt, self.params.linear1.weight, bias=self.params.linear1.bias)
            x = ttnn.relu(x)
            tgt2 = ttnn.linear(x, self.params.linear2.weight, bias=self.params.linear2.bias)

            tgt = ttnn.layer_norm(
                tgt,
                weight=self.params.norm2.weight,
                bias=self.params.norm2.bias,
                epsilon=self.eps,
            )

            if self.update_query_pos:
                x = ttnn.linear(tgt, self.params.linear_pos1.weight, bias=self.params.linear_pos1.bias)
                x = ttnn.relu(x)
                query_pos2 = ttnn.linear(x, self.params.linear_pos2.weight, bias=self.params.linear_pos2.bias)

                query_pos = ttnn.layer_norm(
                    query_pos,
                    weight=self.params.norm_pos.weight,
                    bias=self.params.norm_pos.bias,
                    epsilon=self.eps,
                )

                track_instances.query = ttnn.to_torch(track_instances.query)
                track_instances.query[:, : dim // 2] = ttnn.to_torch(query_pos)
                track_instances.query = ttnn.from_torch(track_instances.query, device=self.device)

            x = ttnn.linear(tgt, self.params.linear_feat1.weight, bias=self.params.linear_feat1.bias)
            x = ttnn.relu(x)
            query_feat2 = ttnn.linear(x, self.params.linear_feat2.weight, bias=self.params.linear_feat2.bias)

            query_feat = ttnn.layer_norm(
                query_feat,
                weight=self.params.norm_feat.weight,
                bias=self.params.norm_feat.bias,
                epsilon=self.eps,
            )

            track_instances.query = ttnn.to_torch(track_instances.query)
            track_instances.query[:, dim // 2 :] = ttnn.to_torch(query_feat)
            track_instances.query = ttnn.from_torch(track_instances.query, device=self.device)

        return track_instances

    def __call__(self, data):
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances = data["init_track_instances"]
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances
