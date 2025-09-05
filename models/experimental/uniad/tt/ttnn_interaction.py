# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.uniad.tt.ttnn_transformer_decoder_layer import TtTransformerDecoderLayer
from models.experimental.uniad.tt.ttnn_transformer_encoder_layer import TtTransformerEncoderLayer


class TtMapInteraction:
    def __init__(
        self,
        parameters,
        device,
        embed_dims=256,
        num_heads=8,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        self.batch_first = batch_first
        self.interaction_transformer = TtTransformerDecoderLayer(
            parameters=parameters.interaction_transformer,
            device=device,
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query, key, query_pos=None, key_pos=None):
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        query = ttnn.reshape(query, (query.shape[0] * query.shape[1], query.shape[2], query.shape[3]))
        mem = ttnn.expand(key, (B * A, -1, -1))
        query = self.interaction_transformer(query, mem)
        ttnn.deallocate(mem)
        query = ttnn.reshape(query, (B, A, P, D))
        return query


class TtTrackAgentInteraction:
    def __init__(
        self,
        parameters,
        device,
        embed_dims=256,
        num_heads=8,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        self.batch_first = batch_first
        self.interaction_transformer = TtTransformerDecoderLayer(
            parameters=parameters.interaction_transformer,
            device=device,
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query, key, query_pos=None, key_pos=None):
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        mem = ttnn.expand(key, (B * A, -1, -1))
        query = ttnn.reshape(query, (query.shape[0] * query.shape[1], query.shape[2], query.shape[3]))
        query = self.interaction_transformer(query, mem)
        ttnn.deallocate(mem)
        query = ttnn.reshape(query, (B, A, P, D))
        return query


class TtIntentionInteraction:
    def __init__(
        self,
        parameters,
        device,
        embed_dims=256,
        num_heads=8,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        self.batch_first = batch_first
        self.interaction_transformer = TtTransformerEncoderLayer(
            parameters=parameters.interaction_transformer,
            device=device,
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query):
        B, A, P, D = query.shape
        rebatch_x = ttnn.reshape(query, (B * A, P, D))
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = ttnn.reshape(rebatch_x, (B, A, P, D))
        return out
