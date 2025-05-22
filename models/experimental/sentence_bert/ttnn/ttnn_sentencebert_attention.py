# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_self_attention import TtnnSentenceBertSelfAttention
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_self_output import TtnnSentenceBertSelfOutput


class TtnnSentenceBertAttention:
    def __init__(self, parameters, config):
        self.self = TtnnSentenceBertSelfAttention(parameters.self, config)
        self.output = TtnnSentenceBertSelfOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        self_outputs = self.self(hidden_states, attention_mask, device=device)
        self_outputs = ttnn.to_memory_config(
            self_outputs,
            memory_config=ttnn.create_sharded_memory_config(
                self_outputs.shape,
                core_grid=device.core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
        )
        self_outputs = self.output(self_outputs, hidden_states)
        return self_outputs
