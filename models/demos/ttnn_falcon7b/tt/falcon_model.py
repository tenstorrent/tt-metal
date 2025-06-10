# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Optional, Tuple

import torch

import ttnn
from models.demos.ttnn_falcon7b.tt.common import create_attention_mask
from models.demos.ttnn_falcon7b.tt.falcon_decoder import TtFalconDecoderLayer
from ttnn import ShardTensorToMesh


class TtFalconModelShared:
    @abstractmethod
    def __init__(
        self,
        device,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        parameters,
    ):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        # So far on CPU until we add embeddings support on device
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.weight = torch.nn.Parameter(parameters.word_embeddings.weight)

        # stack all decoders
        self.layers = [
            TtFalconDecoderLayer(
                device=device,
                config=config,
                model_config=model_config,
                parameters=parameters.h[layer_num],
            )
            for layer_num in range(num_layers)
        ]
        self.ln_f_weight = parameters.ln_f.weight
        self.ln_f_bias = parameters.ln_f.bias
        self.layernorm_eps = config.layer_norm_epsilon

    def model_preprocessing(self, llm_mode, input_ids, kv_cache_len, num_input_tokens):
        assert input_ids.dim() == 2
        batch_size, sequence_size = input_ids.shape

        embeddings = self.embeddings(input_ids)

        if self.device.get_num_devices() == 1:
            mesh_mapper = None
        else:
            shard_dim = 2 if llm_mode == "decode" else 0
            mesh_mapper = ShardTensorToMesh(self.device, dim=shard_dim)

        # Generate input and attention_mask ---------------------------------------------
        if llm_mode == "prefill":
            tt_embeddings = ttnn.from_torch(
                embeddings.unsqueeze(1),
                device=self.device,
                memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        elif llm_mode == "decode":
            assert batch_size % 32 == 0, "For decode, batch_size must be multiple of 32!"
            assert sequence_size == 1, "For decode, q_len must be 1!"

            tt_embeddings = ttnn.from_torch(
                embeddings.unsqueeze(1).transpose(0, 2),
                device=self.device,
                memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        attention_mask, tt_attention_mask = create_attention_mask(
            llm_mode,
            self.model_config["DEFAULT_DTYPE"],
            embeddings,
            batch_size,
            sequence_size,
            self.config.num_attention_heads,
            kv_cache_len,
            self.device,
            mesh_mapper=mesh_mapper,
        )

        return tt_embeddings, tt_attention_mask

    @abstractmethod
    def __call__(
        self,
        input_embeddings: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        layer_output = input_embeddings
        presents = ()
        for idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=layer_output,
                alibi=None,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past[idx],
                layer_past_len=layer_past_len,
                use_cache=use_cache,
            )
            presents += layer_output[1:]
            layer_output = layer_output[0]

        # apply final norm layer
        layer_output = ttnn.layer_norm(
            layer_output,
            epsilon=self.layernorm_eps,
            memory_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        layer_output = ttnn.mul(
            layer_output,
            self.ln_f_weight,
            memory_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
        )
        layer_output = ttnn.add(
            layer_output,
            self.ln_f_bias,
            memory_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
        )

        return layer_output, presents


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        device,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        parameters,
    ):
        super().__init__(
            device=device,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            parameters=parameters,
        )

    def __call__(
        self,
        input_embeddings: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states, presents = super().__call__(
            input_embeddings=input_embeddings,
            llm_mode=llm_mode,
            attention_mask=attention_mask,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )
        return hidden_states, presents
