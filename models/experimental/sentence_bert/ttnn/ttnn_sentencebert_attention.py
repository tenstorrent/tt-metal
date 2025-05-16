# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        self_outputs = self.output(self_outputs, hidden_states)
        return self_outputs
