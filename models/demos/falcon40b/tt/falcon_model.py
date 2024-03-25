# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from abc import abstractmethod
from typing import Optional, Tuple

import tt_lib

from models.demos.falcon40b.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.falcon40b.tt.falcon_attention import generate_cos_sin_cache
from models.utility_functions import (
    torch2tt_tensor,
    pad_by_zero,
    nearest_32,
)


class TtFalconModelShared:
    @abstractmethod
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    ):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.devices = devices
        self.state_dict = state_dict
        self.base_url = base_url
        self.config = config
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config
        self.num_layers = num_layers

        # So far on CPU until we add embeddings support on device
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        word_embeddings_path = tt_cache_path / "embedding.pt"
        if (word_embeddings_path).exists():
            self.embeddings.weight = torch.nn.Parameter(
                torch.load(word_embeddings_path, map_location=torch.device("cpu"))
            )
        else:
            embed_weights = state_dict["transformer.word_embeddings.weight"]
            torch.save(embed_weights, word_embeddings_path)
            self.embeddings.weight = torch.nn.Parameter(embed_weights)

        if use_global_cos_sin_cache:
            global_cos_sin_cache = generate_cos_sin_cache(
                devices,
                config.hidden_size // config.num_attention_heads,
                base_url,
                max_position_embeddings,
                model_config=model_config,
                tt_cache_path=tt_cache_path,
            )
        else:
            global_cos_sin_cache = None

        # stack all decoders
        self.layers = [
            TtFalconDecoderLayer(
                devices=devices,
                state_dict=state_dict,
                base_url=f"{base_url}.h",
                layer_num=layer_num,
                config=config,
                max_position_embeddings=max_position_embeddings,
                model_config=model_config,
                tt_cache_path=tt_cache_path,
                global_cos_sin_cache=global_cos_sin_cache,
            )
            for layer_num in range(num_layers)
        ]

        layer_name = f"{base_url}"

        layernorm_weights_str = f"{layer_name}.ln_f.weight"
        layernorm_bias_str = f"{layer_name}.ln_f.bias"

        layernorm_weights_path = (
            tt_cache_path / f"{layernorm_weights_str}_rm_{self.model_config['LN_F_WEIGHTS_DTYPE'].name}.bin"
        )
        layernorm_bias_path = tt_cache_path / f"{layernorm_bias_str}_rm_{self.model_config['LN_F_BIAS_DTYPE'].name}.bin"

        if (layernorm_weights_path).exists():
            layernorm_gamma_host = tt_lib.tensor.load_tensor(str(layernorm_weights_path))
            self.layernorm_gamma = [
                layernorm_gamma_host.to(device, self.model_config["LN_F_WEIGHTS_MEMCFG"]) for device in devices
            ]
        else:
            layernorm_gamma_host = tt_lib.tensor.Tensor(
                self.state_dict[layernorm_weights_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_F_WEIGHTS_DTYPE"],
            )
            self.layernorm_gamma = [
                layernorm_gamma_host.to(device, self.model_config["LN_F_WEIGHTS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(layernorm_weights_path),
                layernorm_gamma_host,
            )

        if (layernorm_bias_path).exists():
            layernorm_beta_host = tt_lib.tensor.load_tensor(str(layernorm_bias_path))
            self.layernorm_beta = [
                layernorm_beta_host.to(device, self.model_config["LN_F_BIAS_MEMCFG"]) for device in devices
            ]
        else:
            layernorm_beta_host = tt_lib.tensor.Tensor(
                self.state_dict[layernorm_bias_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_F_BIAS_DTYPE"],
            )
            self.layernorm_beta = [
                layernorm_beta_host.to(device, self.model_config["LN_F_BIAS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(layernorm_bias_path),
                layernorm_beta_host,
            )

        self.layernorm_eps = config.layer_norm_epsilon

    def set_model_config(self, model_config):
        self.model_config = model_config
        for layer_num in range(self.num_layers):
            self.layers[layer_num].set_model_config(model_config)

    def model_preprocessing(self, llm_mode, input_ids, kv_cache_len, num_input_tokens):
        assert input_ids.dim() == 2
        batch_size, sequence_size = input_ids.shape

        embeddings = self.embeddings(input_ids)

        # Generate input and attention_mask ---------------------------------------------
        if llm_mode == "prefill":
            assert batch_size == 1, "For prefill, batch_size must be 1!"
            assert sequence_size % 32 == 0, "For prefill, sequence_size must be multiple of 32!"
            assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

            embeddings = torch.chunk(embeddings.unsqueeze(1), len(self.devices), -1)
            tt_embeddings = [
                torch2tt_tensor(
                    embeddings[i],
                    self.devices[i],
                    tt_memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                    tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                )
                for i in range(len(self.devices))
            ]

            attention_mask_bool = torch.ones(batch_size, 1, sequence_size, num_input_tokens, dtype=bool)
            attention_mask_bool = attention_mask_bool.triu(diagonal=1)

            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool,
                    torch.ones(batch_size, 1, sequence_size, sequence_size - num_input_tokens, dtype=bool),
                ),
                dim=-1,
            )

            attention_mask_bool_chunks = torch.chunk(
                (attention_mask_bool_padded * -1e3).expand(-1, self.config.num_attention_heads, -1, -1),
                len(self.devices),
                1,
            )
            tt_attention_mask = []
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = sequence_size
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

            for i in range(len(self.devices)):
                tt_attention_mask.append(
                    torch2tt_tensor(
                        attention_mask_bool_chunks[i],
                        self.devices[i],
                        tt_memory_config=attention_mask_memconfig,
                        tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
                    )
                )

        elif llm_mode == "decode":
            assert batch_size % 32 == 0, "For decode, batch_size must be multiple of 32!"
            assert sequence_size == 1, "For decode, q_len must be 1!"

            embeddings = torch.chunk(embeddings.unsqueeze(1).transpose(0, 2), len(self.devices), -1)
            tt_embeddings = [
                torch2tt_tensor(embeddings[i], None, tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"])
                for i in range(len(self.devices))
            ]

            attention_mask_bool = torch.zeros(batch_size, 1, sequence_size, num_input_tokens, dtype=bool)

            num_max_tokens = nearest_32(
                kv_cache_len + 1
            )  # Potentially, num_max_tokens must be provided as a separate argument
            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool,
                    torch.ones(batch_size, 1, sequence_size, num_max_tokens - num_input_tokens, dtype=bool),
                ),
                dim=-1,
            )
            attention_mask_bool_padded = torch.chunk(
                (attention_mask_bool_padded.transpose(0, 2) * -1e3).expand(-1, self.config.num_attention_heads, -1, -1),
                len(self.devices),
                1,
            )
            tt_attention_mask = [
                torch2tt_tensor(attention_mask_bool_padded[i], None, tt_dtype=self.model_config["ATTN_MASK_DTYPE"])
                for i in range(len(self.devices))
            ]

        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        return tt_embeddings, tt_attention_mask

    @abstractmethod
    def __call__(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
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

        for i in range(len(layer_output)):
            layer_output[i] = tt_lib.tensor.sharded_to_interleaved(
                layer_output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
        layer_output = tt_lib.tensor.all_gather(
            layer_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )
        for i in range(len(layer_output)):
            layer_output[i] = tt_lib.tensor.interleaved_to_sharded(
                layer_output[i], sharded_mem_config=self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"]
            )

        # apply final norm layer
        for i in range(len(layer_output)):
            layer_output[i] = tt_lib.operations.primary.layernorm(
                layer_output[i],
                self.layernorm_eps,
                self.layernorm_gamma[i],
                self.layernorm_beta[i],
                self.model_config["LN_F_OUTPUT_MEMCFG"],
                self.model_config["LN_F_PROGCFG"],
            )

        return layer_output, presents


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    ):
        super().__init__(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            use_global_cos_sin_cache=use_global_cos_sin_cache,
        )

    def __call__(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
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
