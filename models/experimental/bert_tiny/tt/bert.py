# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn
from typing import List, Optional
from models.experimental.bert_tiny.tt.bert_encoder import TtBertencoder
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor, torch_to_tt_tensor_rm


class TtBert(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        device=None,
        mem_config=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.state_dict = state_dict
        self.output_mem_config = mem_config
        self.word_embeddings_weight = ttnn.to_device(
            ttnn.from_torch(state_dict[f"bert.embeddings.word_embeddings.weight"], dtype=ttnn.bfloat16),
            device=self.device,
        )
        self.token_embeddings_weight = ttnn.to_device(
            ttnn.from_torch(state_dict[f"bert.embeddings.token_type_embeddings.weight"], dtype=ttnn.bfloat16),
            device=self.device,
        )
        self.position_embeddings_weight = ttnn.to_device(
            ttnn.from_torch(state_dict[f"bert.embeddings.position_embeddings.weight"], dtype=ttnn.bfloat16),
            device=self.device,
        )

        self.gamma = torch_to_tt_tensor_rm(
            state_dict[f"bert.embeddings.LayerNorm.weight"],
            device=self.device,
            put_on_device=False,
        )

        self.beta = torch_to_tt_tensor_rm(
            state_dict[f"bert.embeddings.LayerNorm.bias"],
            device=self.device,
            put_on_device=False,
        )
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.ln = ttnn.layer_norm
        self.encoder = TtBertencoder(
            config=self.config, state_dict=self.state_dict, device=self.device, mem_config=self.output_mem_config
        )

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        token_type_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values_length: int = 0,
    ):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_ids = ttnn.to_device(
                ttnn.from_torch(position_ids),
                device=self.device,
            )

        embedding = ttnn.embedding(input_ids, self.word_embeddings_weight, layout=ttnn.TILE_LAYOUT)

        if token_type_ids:
            embedding += ttnn.embedding(token_type_ids, self.token_embeddings_weight, layout=ttnn.TILE_LAYOUT)

        embedding += ttnn.embedding(position_ids, self.position_embeddings_weight, layout=ttnn.TILE_LAYOUT)

        embedding = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(embedding, layout=ttnn.ROW_MAJOR_LAYOUT)))
        embedding = torch_to_tt_tensor_rm(embedding, self.device, put_on_device=False)
        ln_out = self.ln(embedding, epsilon=1e-12, weight=self.gamma, bias=self.beta)
        ln_out = tt_to_torch_tensor(ln_out)
        encoder_input = ttnn.to_device(ttnn.from_torch(ln_out, dtype=ttnn.bfloat16), device=self.device)
        return self.encoder(encoder_input, attention_mask)
