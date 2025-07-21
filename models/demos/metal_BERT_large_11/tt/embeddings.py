# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn
from models.demos.metal_BERT_large_11.tt.tensor_utils import load_or_compute_and_cache


class TtEmbeddings:
    def __init__(self, hugging_face_reference_model, device, model_config, tt_cache_path):
        self.device = device
        self.model_config = model_config
        config = hugging_face_reference_model.config
        state_dict = hugging_face_reference_model.state_dict()
        self.embedding_dim = config.hidden_size
        self.pad_token = config.pad_token_id

        base_address = "bert.embeddings"

        word_embeddings_path = None
        position_embeddings_path = None
        token_type_embeddings_path = None
        layerNorm_gamma_path = None
        layerNorm_beta_path = None

        if tt_cache_path is not None:
            word_embeddings_path = str(
                f"{tt_cache_path}/"
                f"{base_address}.word_embeddings.weight_{self.model_config['INPUT_EMBEDDINGS_WEIGHTS_DTYPE'].name}.bin"
            )
            position_embeddings_path = str(
                f"{tt_cache_path}/"
                f"{base_address}.position_embeddings.weight_{self.model_config['INPUT_EMBEDDINGS_WEIGHTS_DTYPE'].name}.bin"
            )
            token_type_embeddings_path = str(
                f"{tt_cache_path}/"
                f"{base_address}.token_type_embeddings.weight_{self.model_config['INPUT_EMBEDDINGS_WEIGHTS_DTYPE'].name}.bin"
            )
            layerNorm_gamma_path = str(
                f"{tt_cache_path}/"
                f"{base_address}.LayerNorm.weight_{self.model_config['EMBEDDINGS_LAYERNORM_GAMMA_DTYPE'].name}.bin"
            )
            layerNorm_beta_path = str(
                f"{tt_cache_path}/"
                f"{base_address}.LayerNorm.beta_{self.model_config['EMBEDDINGS_LAYERNORM_BETA_DTYPE'].name}.bin"
            )

        def compute_word_embeddings():
            weight_torch = state_dict[f"{base_address}.word_embeddings.weight"]
            return ttnn.from_torch(
                weight_torch,
                dtype=model_config["INPUT_EMBEDDINGS_WEIGHTS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_position_embeddings():
            weight_torch = state_dict[f"{base_address}.position_embeddings.weight"]
            return ttnn.from_torch(
                weight_torch,
                dtype=model_config["INPUT_EMBEDDINGS_WEIGHTS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_token_type_embeddings():
            weight_torch = state_dict[f"{base_address}.token_type_embeddings.weight"]
            return ttnn.from_torch(
                weight_torch,
                dtype=model_config["INPUT_EMBEDDINGS_WEIGHTS_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_layerNorm_gamma():
            gamma_torch = state_dict[f"{base_address}.LayerNorm.weight"].reshape([1, 1, -1, 32])
            return ttnn.from_torch(
                gamma_torch,
                dtype=model_config["EMBEDDINGS_LAYERNORM_GAMMA_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_layerNorm_beta():
            beta_torch = state_dict[f"{base_address}.LayerNorm.bias"].reshape([1, 1, -1, 32])
            return ttnn.from_torch(
                beta_torch,
                dtype=model_config["EMBEDDINGS_LAYERNORM_BETA_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self.word_embeddings_weight = load_or_compute_and_cache(
            word_embeddings_path,
            compute_word_embeddings,
            device=device,
            mem_config=self.model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"],
        )
        self.position_embeddings_weight = load_or_compute_and_cache(
            position_embeddings_path,
            compute_position_embeddings,
            device=device,
            mem_config=self.model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"],
        )
        self.token_type_embeddings_weight = load_or_compute_and_cache(
            token_type_embeddings_path,
            compute_token_type_embeddings,
            device=device,
            mem_config=self.model_config["INPUT_EMBEDDINGS_WEIGHTS_MEMCFG"],
        )
        self.layerNorm_gamma = load_or_compute_and_cache(
            layerNorm_gamma_path,
            compute_layerNorm_gamma,
            device=device,
            mem_config=self.model_config["EMBEDDINGS_LAYERNORM_GAMMA_MEMCFG"],
        )
        self.layerNorm_beta = load_or_compute_and_cache(
            layerNorm_beta_path,
            compute_layerNorm_beta,
            device=device,
            mem_config=self.model_config["EMBEDDINGS_LAYERNORM_BETA_MEMCFG"],
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
        input_tt_tensor = ttnn.Tensor(input_ids_torch, ttnn.uint32)
        token_type_ids_torch = torch.reshape(token_type_ids, input_ids_shape)
        token_type_ids_tt_tensor = ttnn.Tensor(token_type_ids_torch, ttnn.uint32)
        position_ids_tt_tensor = None
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if self.position_embedding_type == "absolute":
            position_ids_torch = torch.reshape(position_ids, (1, 1, seq_length))
            position_ids_torch = position_ids_torch.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            position_ids_tt_tensor = ttnn.Tensor(position_ids_torch, ttnn.uint32)
        return {
            "input_ids": input_tt_tensor,
            "token_type_ids": token_type_ids_tt_tensor,
            "position_ids": position_ids_tt_tensor,
        }

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        token_type_ids: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        inputs_embeds = ttnn.embedding(
            input_ids,
            self.word_embeddings_weight,
            layout=ttnn.TILE_LAYOUT,
            embeddings_type=ttnn.EmbeddingsType.PADDED,
            padding_idx=self.pad_token,
            memory_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"],
        )
        input_embeds = ttnn.reshape(
            inputs_embeds, [inputs_embeds.shape[0], 1, inputs_embeds.shape[1], inputs_embeds.shape[2]]
        )
        input_ids.deallocate()

        token_type_embeddings = ttnn.embedding(
            token_type_ids,
            self.token_type_embeddings_weight,
            layout=ttnn.TILE_LAYOUT,
            embeddings_type=ttnn.EmbeddingsType.BINARY,
            memory_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"],
        )
        token_type_ids.deallocate()

        if self.position_embedding_type == "absolute":
            inputs_plus_token_type_embeddings_tt_tensor = ttnn.add(
                inputs_embeds, token_type_embeddings, memory_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"]
            )
            if not self.model_config["DEALLOC_INPUT_EMBEDS_AFTER_POSITION_EMBEDS"]:
                inputs_embeds.deallocate()
                token_type_embeddings.deallocate()

            position_embeddings_tt_tensor = ttnn.embedding(
                position_ids,
                self.position_embeddings_weight,
                layout=ttnn.TILE_LAYOUT,
                embeddings_type=ttnn.EmbeddingsType.GENERIC,
                memory_config=self.model_config["OUTPUT_EMBEDDINGS_MEMCFG"],
            )
            position_embeddings_tt_tensor = ttnn.reshape(
                position_embeddings_tt_tensor,
                [
                    position_embeddings_tt_tensor.shape[0],
                    1,
                    position_embeddings_tt_tensor.shape[1],
                    position_embeddings_tt_tensor.shape[2],
                ],
            )
            inputs_plus_token_type_embeddings_tt_tensor = ttnn.reshape(
                inputs_plus_token_type_embeddings_tt_tensor,
                [
                    inputs_plus_token_type_embeddings_tt_tensor.shape[0],
                    1,
                    inputs_plus_token_type_embeddings_tt_tensor.shape[1],
                    inputs_plus_token_type_embeddings_tt_tensor.shape[2],
                ],
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
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            )
        else:
            embeddings_tt_tensor_layerNorm = ttnn.layer_norm(
                inputs_embeds,
                residual_input_tensor=token_type_embeddings,
                epsilon=self.layerNorm_eps,
                weight=self.layerNorm_gamma,
                bias=self.layerNorm_beta,
                memory_config=self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"],
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            )
        return embeddings_tt_tensor_layerNorm
