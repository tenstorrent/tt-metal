# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_layer import TtnnSentenceBertLayer


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
        if attention_mask.is_sharded():
            attention_mask_interleaved = ttnn.sharded_to_interleaved(attention_mask, ttnn.L1_MEMORY_CONFIG)
            attention_mask_interleaved = ttnn.to_layout(attention_mask_interleaved, ttnn.TILE_LAYOUT)
            ttnn.deallocate(attention_mask)
        else:
            attention_mask_interleaved = attention_mask
        for i in range(len(self.layers)):
            layer_outputs = self.layers[i](hidden_states, attention_mask_interleaved, device=device)
            hidden_states = layer_outputs
        return hidden_states
