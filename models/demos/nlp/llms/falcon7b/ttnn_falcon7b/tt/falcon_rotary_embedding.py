# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn


def falcon_rotary_embedding(
    layer: ttnn.Tensor, max_position_embeddings, memory_config, *, parameters, token_idx: Optional[int] = None
) -> ttnn.Tensor:
    seq_len = layer.shape[2]
    assert seq_len <= max_position_embeddings, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

    return ttnn.experimental.rotary_embedding(
        layer, parameters.cos_cached, parameters.sin_cached, token_idx, memory_config=memory_config
    )


class TtFalconRotaryEmbedding:
    def __init__(
        self,
        parameters,
        model_config,
        max_position_embeddings=2048,
    ):
        super().__init__()
        self.parameters = parameters
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

    def __call__(self, layer: ttnn.Tensor, token_idx: Optional[int] = None) -> ttnn.Tensor:
        return falcon_rotary_embedding(
            layer,
            self.max_position_embeddings,
            memory_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
            parameters=self.parameters,
            token_idx=token_idx,
        )
