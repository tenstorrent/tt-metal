# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.uniad.tt.ttnn_transformer_decoder_layer import TtTransformerDecoderLayer
from models.experimental.uniad.tt.ttnn_transformer_encoder_layer import TtTransformerEncoderLayer


class TtMapInteraction:
    """
    Modeling the interaction between the agent and the map
    """

    def __init__(
        self,
        parameters,
        device,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
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
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query, key, query_pos=None, key_pos=None):
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # N, A, P, D -> N*A, P, D
        query = ttnn.reshape(query, (query.shape[0] * query.shape[1], query.shape[2], query.shape[3]))
        # mem = key.expand(B * A, -1, -1)
        mem = ttnn.clone(key)
        query = self.interaction_transformer(query, mem)
        query = ttnn.reshape(query, (B, A, P, D))
        return query


class TtTrackAgentInteraction:
    """
    Modeling the interaction between the agents
    """

    def __init__(
        self,
        parameters,
        device,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
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
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query, key, query_pos=None, key_pos=None):
        """
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        """
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # mem = key.expand(B * A, -1, -1)
        mem = ttnn.clone(key)
        # N, A, P, D -> N*A, P, D
        query = ttnn.reshape(query, (query.shape[0] * query.shape[1], query.shape[2], query.shape[3]))
        query = self.interaction_transformer(query, mem)
        query = ttnn.reshape(query, (B, A, P, D))
        return query


class TtIntentionInteraction:
    """
    Modeling the interaction between anchors
    """

    def __init__(
        self,
        parameters,
        device,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
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
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def __call__(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A,P, D
        rebatch_x = ttnn.reshape(query, (B * A, P, D))
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = ttnn.reshape(rebatch_x, (B, A, P, D))
        return out
