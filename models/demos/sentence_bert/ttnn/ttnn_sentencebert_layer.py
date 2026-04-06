# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_attention import TtnnSentenceBertAttention
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_intermediate import TtnnSentenceBertIntermediate
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_output import TtnnSentenceBertOutput


class TtnnSentenceBertLayer:
    def __init__(self, parameters, config):
        self.attention = TtnnSentenceBertAttention(parameters.attention, config)
        self.intermediate = TtnnSentenceBertIntermediate(parameters.intermediate)
        self.output = TtnnSentenceBertOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, device=device)
        ttnn.deallocate(hidden_states)
        intermediate_output = self.intermediate(self_attention_outputs)
        intermediate_output = self.output(intermediate_output, self_attention_outputs)
        return intermediate_output
