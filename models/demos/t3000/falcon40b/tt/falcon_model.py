# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from abc import abstractmethod
from typing import Optional, Tuple
from tqdm import tqdm

import ttnn

from ttnn import ReplicateTensorToMesh, ShardTensorToMesh
from models.demos.t3000.falcon40b.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.t3000.falcon40b.tt.falcon_embeddings import TtFalconEmbeddings
from models.demos.t3000.falcon40b.tt.falcon_attention import generate_cos_sin_cache
from models.utility_functions import nearest_32

from models.demos.t3000.falcon40b.tt.model_utils import (
    partial_layernorm,
    generate_layernorm_persistent_tensors,
)


class TtFalconModelShared:
    @abstractmethod
    def __init__(
        self,
        device_mesh,
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
        self.device_mesh = device_mesh
        self.state_dict = state_dict
        self.base_url = base_url
        self.config = config
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config
        self.num_layers = num_layers
        self.hidden_size = config.hidden_size
        self.num_devices = device_mesh.get_num_devices()
        self.ln_output_tensors_dict = {
            "final_layernorm": dict(),
            "mlp_layernorm": dict(),
            "attn_layernorm": dict(),
        }

        # Word Embeddings
        self.embeddings = TtFalconEmbeddings(
            device_mesh=device_mesh,
            state_dict=state_dict,
            cache_path=tt_cache_path,
            model_config=model_config,
        )

        if use_global_cos_sin_cache:
            global_cos_sin_cache = generate_cos_sin_cache(
                device_mesh,
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
                device_mesh=device_mesh,
                state_dict=state_dict,
                base_url=f"{base_url}.h",
                layer_num=layer_num,
                config=config,
                max_position_embeddings=max_position_embeddings,
                model_config=model_config,
                tt_cache_path=tt_cache_path,
                global_cos_sin_cache=global_cos_sin_cache,
                ln_output_tensors_dict=self.ln_output_tensors_dict,
            )
            for layer_num in tqdm(range(num_layers), desc="Loading decoder layers")
        ]

        layer_name = f"{base_url}"

        layernorm_weights_str = f"{layer_name}.ln_f.weight"
        layernorm_bias_str = f"{layer_name}.ln_f.bias"

        layernorm_weights_path = (
            tt_cache_path / f"{layernorm_weights_str}_rm_{self.model_config['LN_F_WEIGHTS_DTYPE'].name}"
        )
        layernorm_bias_path = tt_cache_path / f"{layernorm_bias_str}_rm_{self.model_config['LN_F_BIAS_DTYPE'].name}"

        self.layernorm_gamma = ttnn.as_tensor(
            tensor=self.state_dict[layernorm_weights_str],
            dtype=self.model_config["LN_F_WEIGHTS_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device_mesh,
            memory_config=self.model_config["LN_F_WEIGHTS_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
            cache_file_name=layernorm_weights_path,
            preprocess=lambda x: x.reshape(1, 1, -1, 32),
        )

        self.layernorm_beta = ttnn.as_tensor(
            tensor=self.state_dict[layernorm_bias_str],
            dtype=self.model_config["LN_F_BIAS_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device_mesh,
            memory_config=self.model_config["LN_F_BIAS_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
            cache_file_name=layernorm_bias_path,
            preprocess=lambda x: x.reshape(1, 1, -1, 32),
        )

        self.layernorm_eps = config.layer_norm_epsilon
        # push attention_mask to device in row major order and then tilize on device (faster than tilizing on CPU)
        self.max_attn_mask = self.create_attn_mask(max_position_embeddings)

    def create_attn_mask(self, max_seq_len):
        attn_mask_bool = torch.ones(1, 1, max_seq_len, max_seq_len, dtype=bool)
        attn_mask_bool = attn_mask_bool.triu(diagonal=1)
        attention_mask_memconfig = ttnn.DRAM_MEMORY_CONFIG

        tt_attn_mask = ttnn.as_tensor(
            tensor=attn_mask_bool,
            dtype=self.model_config["BFLOAT16_DTYPE"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=attention_mask_memconfig,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            preprocess=lambda x: (x * -1e5),
        )

        tt_attn_mask = ttnn.tilize(
            tt_attn_mask,
            memory_config=attention_mask_memconfig,
            dtype=self.model_config["ATTN_MASK_DTYPE"],
        )
        return tt_attn_mask

    def slice_attn_mask(self, seq_len):
        assert seq_len % 32 == 0, "seq_len must be multiple of 32!"
        sliced_attn_mask = self.max_attn_mask[:, :, :seq_len, :seq_len]
        return sliced_attn_mask

    def initialize_kv_cache(self):
        layer_past = ()
        for layer_num in range(self.num_layers):
            layer_past += self.layers[layer_num].self_attn.initialize_kvcache()
        return layer_past

    def set_model_config(self, model_config):
        self.model_config = model_config
        self.embeddings.set_model_config(model_config)
        for layer_num in range(self.num_layers):
            self.layers[layer_num].set_model_config(model_config)

    def model_preprocessing(self, llm_mode, input_ids, kv_cache_len, num_input_tokens):
        assert input_ids.dim() == 2
        batch_size, sequence_size = input_ids.shape

        if llm_mode == "decode":
            input_ids = input_ids.reshape(sequence_size, 1, 1, batch_size)
        else:
            input_ids = input_ids.reshape(batch_size, 1, 1, sequence_size)

        tt_inputs = ttnn.as_tensor(
            tensor=input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )

        # Generate input and attention_mask ---------------------------------------------
        if llm_mode == "prefill":
            assert batch_size == 1, "For prefill, batch_size must be 1!"
            assert sequence_size % 32 == 0, "For prefill, sequence_size must be multiple of 32!"
            assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

            tt_attention_mask = self.slice_attn_mask(sequence_size)

            # Genereate ln output tensors for prefill if not existing
            do_generate_ln_tensors = (
                sequence_size > self.model_config["layernorm_params"]["slice_size"]
                and sequence_size not in self.ln_output_tensors_dict["final_layernorm"]
            )
            if do_generate_ln_tensors:
                generate_layernorm_persistent_tensors(
                    sequence_size,
                    self.model_config["layernorm_params"]["slice_size"],
                    self.ln_output_tensors_dict,
                    self.device_mesh,
                    self.hidden_size,
                    self.model_config["LN_MLP_OUTPUT_DTYPE"],
                )
        elif llm_mode == "decode":
            assert batch_size % 32 == 0, "For decode, batch_size must be multiple of 32!"
            assert sequence_size == 1, "For decode, q_len must be 1!"

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

            # Push attention mask to device in row major order and then tilize on device (faster than tilizing on CPU)
            tt_attention_mask = ttnn.as_tensor(
                tensor=attention_mask_bool_padded,
                dtype=self.model_config["BFLOAT16_DTYPE"],  # subsequent tilize op expects bfloat16 inputs
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device_mesh,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                preprocess=lambda x: (x.transpose(0, 2) * -1e5).expand(1, 1, -1, -1),
            )

            tt_attention_mask = ttnn.tilize(
                tt_attention_mask,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
                dtype=self.model_config["ATTN_MASK_DTYPE"],
            )

        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        return tt_inputs, tt_attention_mask

    @abstractmethod
    def __call__(
        self,
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        input_embeddings = self.embeddings(input_ids)

        if llm_mode == "prefill":
            return self.fwd_prefill(
                input_embeddings=input_embeddings,
                llm_mode=llm_mode,
                attention_mask=attention_mask,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                use_cache=use_cache,
            )
        elif llm_mode == "decode":
            return self.fwd_decode(
                input_embeddings=input_embeddings,
                llm_mode=llm_mode,
                attention_mask=attention_mask,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                use_cache=use_cache,
            )
        else:
            assert False

    def fwd_prefill(
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

        if layer_output.dtype != self.model_config["BFP8_DTYPE"]:
            layer_output = ttnn.experimental.typecast(
                layer_output, self.model_config["BFP8_DTYPE"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        layer_output = ttnn.all_gather(
            layer_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )

        if self.model_config["LN_INPUT_DTYPE"] != self.model_config["BFP8_DTYPE"]:
            layer_output = ttnn.experimental.typecast(
                layer_output, self.model_config["LN_INPUT_DTYPE"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # apply final norm layer
        layer_output = partial_layernorm(
            layer_output,
            self.layernorm_gamma,
            self.layernorm_beta,
            self.layernorm_eps,
            self.model_config["layernorm_params"],
            self.model_config["PARTIAL_LN_MEMCFG"],
            self.model_config["PARTIAL_LN_INPLACE_PROGCFG"],
            self.model_config["LN_MLP_OUTPUT_DTYPE"],
            self.ln_output_tensors_dict["final_layernorm"],
        )

        return layer_output, presents

    def fwd_decode(
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

        layer_output = ttnn.sharded_to_interleaved(
            layer_output,
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        layer_output = ttnn.all_gather(
            layer_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        layer_output = ttnn.interleaved_to_sharded(
            layer_output,
            self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # apply final norm layer
        layer_output = ttnn.layer_norm(
            layer_output,
            epsilon=self.layernorm_eps,
            weight=self.layernorm_gamma,
            bias=self.layernorm_beta,
            memory_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
            program_config=self.model_config["LN_F_PROGCFG"],
        )

        return layer_output, presents


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        device_mesh,
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
            device_mesh=device_mesh,
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
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states, presents = super().__call__(
            input_ids=input_ids,
            llm_mode=llm_mode,
            attention_mask=attention_mask,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )
        return hidden_states, presents
