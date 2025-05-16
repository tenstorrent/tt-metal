# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_layer import TtnnSentenceBertLayer


class TtnnSentenceBertEncoder:
    def __init__(self, parameters, config):
        self.layers = {}
        for i in range(config.num_hidden_layers):
            self.layers[i] = TtnnSentenceBertLayer(parameters.layer[i], config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        for i in range(len(self.layers)):
            layer_outputs = self.layers[i](hidden_states, attention_mask, device=device)
            hidden_states = layer_outputs
        return hidden_states
