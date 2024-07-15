# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union
import torch
import ttnn
import tt_lib as ttl
from models.utility_functions import torch2tt_tensor


class TtEmbeddings:
    def __init__(self, hugging_face_reference_model, device, model_config, tt_cache_path):
        self.device = device
        self.model_config = model_config
        config = hugging_face_reference_model.config
        state_dict = hugging_face_reference_model.state_dict()
        self.embedding_dim = config.hidden_size
        self.pad_token = config.pad_token_id

        base_address = "bert.embeddings"
        if tt_cache_path is not None:
            self.word_embeddings_weight = ttl.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{base_address}.word_embeddings.weight_{self.model_config['INPUT_EMBEDDINGS_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"])
            self.position_embeddings_weight = ttl.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{base_address}.position_embeddings.weight_{self.model_config['INPUT_EMBEDDINGS_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"])
            self.token_type_embeddings_weight = ttl.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{base_address}.token_type_embeddings.weight_{self.model_config['INPUT_EMBEDDINGS_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"])
            self.layerNorm_gamma = ttl.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{base_address}.LayerNorm.weight_{self.model_config['EMBEDDINGS_LAYERNORM_GAMMA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["EMBEDDINGS_LAYERNORM_GAMMA_MEMCFG"])
            self.layerNorm_beta = ttl.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{base_address}.LayerNorm.beta_{self.model_config['EMBEDDINGS_LAYERNORM_BETA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["EMBEDDINGS_LAYERNORM_BETA_MEMCFG"])
        else:
            self.word_embeddings_weight = torch2tt_tensor(
                state_dict[f"{base_address}.word_embeddings.weight"],
                device,
                ttl.tensor.Layout.ROW_MAJOR,
                model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"],
                model_config["INPUT_EMBEDDINGS_WEIGHTS_DTYPE"],
            )

            self.position_embeddings_weight = torch2tt_tensor(
                state_dict[f"{base_address}.position_embeddings.weight"],
                device,
                ttl.tensor.Layout.ROW_MAJOR,
                model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"],
                model_config["INPUT_EMBEDDINGS_WEIGHTS_DTYPE"],
            )

            self.token_type_embeddings_weight = torch2tt_tensor(
                state_dict[f"{base_address}.token_type_embeddings.weight"],
                device,
                ttl.tensor.Layout.ROW_MAJOR,
                model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"],
                model_config["INPUT_EMBEDDINGS_WEIGHTS_DTYPE"],
            )

            self.layerNorm_gamma = torch2tt_tensor(
                state_dict[f"{base_address}.LayerNorm.weight"].reshape([1, 1, -1, 32]),
                device,
                ttl.tensor.Layout.ROW_MAJOR,
                model_config["EMBEDDINGS_LAYERNORM_GAMMA_MEMCFG"],
                model_config["EMBEDDINGS_LAYERNORM_GAMMA_DTYPE"],
            )

            self.layerNorm_beta = torch2tt_tensor(
                state_dict[f"{base_address}.LayerNorm.bias"].reshape([1, 1, -1, 32]),
                device,
                ttl.tensor.Layout.ROW_MAJOR,
                model_config["EMBEDDINGS_LAYERNORM_BETA_MEMCFG"],
                model_config["EMBEDDINGS_LAYERNORM_BETA_DTYPE"],
            )

        self.layerNorm_eps = config.layer_norm_eps

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.token_type_ids = torch.zeros(self.position_ids.size(), dtype=torch.long)

    def preprocess_embedding_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ):
        input_shape = input_ids.size()

        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_ids_shape = (batch_size, 1, 1, seq_length)
        input_ids_torch = torch.reshape(input_ids, input_ids_shape)
        input_tt_tensor = ttl.tensor.Tensor(input_ids_torch, ttl.tensor.DataType.UINT32)
        token_type_ids_torch = torch.reshape(token_type_ids, input_ids_shape)
        token_type_ids_tt_tensor = ttl.tensor.Tensor(token_type_ids_torch, ttl.tensor.DataType.UINT32)
        position_ids_tt_tensor = None
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if self.position_embedding_type == "absolute":
            position_ids_torch = torch.reshape(position_ids, (1, 1, seq_length))
            position_ids_torch = position_ids_torch.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            position_ids_tt_tensor = ttl.tensor.Tensor(position_ids_torch, ttl.tensor.DataType.UINT32)
        return {
            "input_ids": input_tt_tensor,
            "token_type_ids": token_type_ids_tt_tensor,
            "position_ids": position_ids_tt_tensor,
        }

    def __call__(
        self,
        input_ids: ttl.tensor.Tensor,
        token_type_ids: ttl.tensor.Tensor,
        position_ids: Optional[ttl.tensor.Tensor] = None,
    ) -> ttl.tensor.Tensor:
        inputs_embeds = ttl.tensor.embeddings(
            input_ids,
            self.word_embeddings_weight,
            tilized=True,
            embeddings_type=ttl.tensor.EmbeddingsType.PADDED,
            pad_token=self.pad_token,
            output_mem_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"],
        )
        input_ids.deallocate()

        token_type_embeddings = ttl.tensor.embeddings(
            token_type_ids,
            self.token_type_embeddings_weight,
            tilized=True,
            embeddings_type=ttl.tensor.EmbeddingsType.BINARY,
            output_mem_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"],
        )
        token_type_ids.deallocate()

        if self.position_embedding_type == "absolute":
            inputs_plus_token_type_embeddings_tt_tensor = ttnn.add(
                inputs_embeds, token_type_embeddings, memory_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"]
            )
            if not self.model_config["DEALLOC_INPUT_EMBEDS_AFTER_POSITION_EMBEDS"]:
                inputs_embeds.deallocate()
                token_type_embeddings.deallocate()

            position_embeddings_tt_tensor = ttl.tensor.embeddings(
                position_ids,
                self.position_embeddings_weight,
                tilized=True,
                embeddings_type=ttl.tensor.EmbeddingsType.GENERIC,
                output_mem_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"],
            )
            # Deallocate inputs_embeds and token_type_embeddings here to avoid having to move final output
            if self.model_config["DEALLOC_INPUT_EMBEDS_AFTER_POSITION_EMBEDS"]:
                inputs_embeds.deallocate()
                token_type_embeddings.deallocate()
            position_ids.deallocate()

            embeddings_tt_tensor_layerNorm = ttnn.layer_norm(
                position_embeddings_tt_tensor,
                residual_input_tensor=inputs_plus_token_type_embeddings_tt_tensor,
                epsilon=self.layerNorm_eps,
                weight=self.layerNorm_gamma,
                bias=self.layerNorm_beta,
                memory_config=self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"],
            )
        else:
            embeddings_tt_tensor_layerNorm = ttnn.layer_norm(
                inputs_embeds,
                residual_input_tensor=token_type_embeddings,
                epsilon=self.layerNorm_eps,
                weight=self.layerNorm_gamma,
                bias=self.layerNorm_beta,
                memory_config=self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"],
            )
        return embeddings_tt_tensor_layerNorm
