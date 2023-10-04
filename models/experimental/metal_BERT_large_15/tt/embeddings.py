# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union
import torch
import tt_lib as ttl
from models.utility_functions import torch_to_tt_tensor_rm


class TtEmbeddings(torch.nn.Module):
    def __init__(self, hugging_face_reference_model, device, input_mem_config, output_mem_config):
        super().__init__()

        self.device = device
        config = hugging_face_reference_model.config
        state_dict = hugging_face_reference_model.state_dict()
        self.embedding_dim = config.hidden_size

        base_address = "bert.embeddings"
        self.word_embeddings_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.word_embeddings.weight"],
            device,
            put_on_device=True,
        )
        self.position_embeddings_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.position_embeddings.weight"],
            device,
            put_on_device=True,
        )
        self.token_type_embeddings_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.token_type_embeddings.weight"],
            device,
            put_on_device=True,
        )
        self.layerNorm_gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.LayerNorm.weight"],
            device,
            put_on_device=True,
        )
        self.layerNorm_beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.LayerNorm.bias"],
            device,
            put_on_device=True,
        )
        self.layerNorm_eps = config.layer_norm_eps

        self.input_mem_config = input_mem_config
        self.output_mem_config = output_mem_config

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        # Disable dropout
        self.eval()

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        input_shape = input_ids.size()

        batch_size = input_shape[0]
        seq_length = input_shape[1]

        input_ids_shape = (batch_size, 1, seq_length, 1)
        input_ids_torch = torch.reshape(input_ids, input_ids_shape)
        input_tt_tensor = ttl.tensor.Tensor(input_ids_torch, ttl.tensor.DataType.UINT32).to(
            self.device, self.input_mem_config
        )
        inputs_embeds = ttl.tensor.embeddings(
            input_tt_tensor,
            self.word_embeddings_weight,
            split_weights=False,
            tilized=True,
            output_mem_config=self.input_mem_config,
        )

        token_type_ids_torch = torch.reshape(token_type_ids, input_ids_shape)
        token_type_ids_tt_tensor = ttl.tensor.Tensor(token_type_ids_torch, ttl.tensor.DataType.UINT32).to(
            self.device, self.input_mem_config
        )
        token_type_embeddings = ttl.tensor.embeddings(
            token_type_ids_tt_tensor,
            self.token_type_embeddings_weight,
            split_weights=False,
            tilized=True,
            output_mem_config=self.input_mem_config,
        )

        inputs_plus_token_type_embeddings_tt_tensor = ttl.tensor.add(inputs_embeds, token_type_embeddings)
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if self.position_embedding_type == "absolute":
            position_ids_torch = torch.reshape(position_ids, (1, seq_length, 1))
            position_ids_torch = position_ids_torch.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            position_ids_tt_tensor = ttl.tensor.Tensor(position_ids_torch, ttl.tensor.DataType.UINT32).to(
                self.device, self.input_mem_config
            )
            position_embeddings_tt_tensor = ttl.tensor.embeddings(
                position_ids_tt_tensor,
                self.position_embeddings_weight,
                split_weights=False,
                tilized=True,
                output_mem_config=self.input_mem_config,
            )
            inputs_plus_token_type_embeddings_tt_tensor = ttl.tensor.add(
                position_embeddings_tt_tensor,
                inputs_plus_token_type_embeddings_tt_tensor,
                output_mem_config=self.input_mem_config,
            )

        embeddings_tt_tensor_layerNorm = ttl.tensor.layernorm(
            inputs_plus_token_type_embeddings_tt_tensor,
            self.layerNorm_eps,
            gamma=self.layerNorm_gamma,
            beta=self.layerNorm_beta,
            output_mem_config=self.output_mem_config,
        )
        return embeddings_tt_tensor_layerNorm


class PytorchEmbeddings(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.bert.embeddings

        # Disable dropout
        self.eval()

    def forward(self, input_ids, token_type_ids=None):
        return self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)


def run_embeddings_inference():
    return
