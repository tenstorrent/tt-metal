# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Optional, Tuple

import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ShardTensorToMesh

from models.demos.falcon7b_common.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.falcon7b_common.tt.model_utils import get_weights_cached, layernorm
from models.utility_functions import nearest_32
from models.demos.falcon7b_common.tests.test_utils import (
    create_prefill_attn_mask_for_sharded_softmax,
    tt_from_torch,
    get_num_devices,
    dump_device_profiler,
)
from tqdm import tqdm


class TtFalconModelShared(torch.nn.Module):
    @abstractmethod
    def __init__(
        self,
        mesh_device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.mesh_device = mesh_device
        self.num_devices = get_num_devices(mesh_device)
        self.state_dict = state_dict
        self.base_url = base_url
        self.config = config
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        layer_name = f"{base_url}"
        embedding_weights_str = f"{layer_name}.word_embeddings.weight"
        self.embedding_weights = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            embedding_weights_str,
            weight_config_str="WORD_EMBEDDING_WEIGHTS",
            weights_to_cache=(state_dict[embedding_weights_str] if state_dict else None),
            tt_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # stack all decoders
        self.layers = torch.nn.ModuleList(
            [
                TtFalconDecoderLayer(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    base_url=f"{base_url}.h",
                    layer_num=layer_num,
                    config=config,
                    max_position_embeddings=max_position_embeddings,
                    model_config=model_config,
                    tt_cache_path=tt_cache_path,
                )
                for layer_num in tqdm(range(num_layers), desc="Loading decoder layers")
            ]
        )

        layernorm_weights_str = f"{layer_name}.ln_f.weight"
        layernorm_bias_str = f"{layer_name}.ln_f.bias"

        self.layernorm_gamma = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            layernorm_weights_str,
            weight_config_str="LN_F_WEIGHTS",
            weights_to_cache=(self.state_dict[layernorm_weights_str] if self.state_dict else None),
        )
        self.layernorm_beta = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            layernorm_bias_str,
            weight_config_str="LN_F_BIAS",
            weights_to_cache=(self.state_dict[layernorm_bias_str] if self.state_dict else None),
        )

        self.layernorm_eps = config.layer_norm_epsilon

    def model_preprocessing(self, llm_mode, input_ids, kv_cache_len, num_input_tokens):
        assert input_ids.dim() == 2
        global_batch_size, sequence_size = input_ids.shape
        batch_size = global_batch_size // self.num_devices

        # Generate input and attention_mask ---------------------------------------------
        if llm_mode == "prefill":
            assert batch_size == 1, "For prefill, batch_size must be 1!"
            assert sequence_size % 32 == 0, "For prefill, sequence_size must be multiple of 32!"
            assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

            # Prepare attention mask
            attention_mask_bool = torch.ones(batch_size, 1, sequence_size, num_input_tokens, dtype=bool)
            attention_mask_bool = attention_mask_bool.triu(diagonal=1)

            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool,
                    torch.ones(batch_size, 1, sequence_size, sequence_size - num_input_tokens, dtype=bool),
                ),
                dim=-1,
            )

            if self.model_config["PREFILL_OPTIMIZED_MODE"] and num_input_tokens in [128, 1024, 2048]:
                attention_mask_ = create_prefill_attn_mask_for_sharded_softmax(
                    attention_mask_bool_padded * -1e5,
                    self.config.num_attention_heads,
                    num_input_tokens,
                )
                # Send attn masks to device
                attn_masks_unordered = [
                    tt_from_torch(
                        attention_mask_slice,
                        dtype=ttnn.bfloat16,  # subsequent tilize op excepts bfloat16 inputs
                        device=self.mesh_device,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                        mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
                    )
                    for attention_mask_slice in attention_mask_
                ]
                # Tilize attn masks
                tt_attention_mask = [
                    ttnn.tilize(
                        tt_attention_mask_slice,
                        memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                        dtype=self.model_config["ATTN_MASK_OPTIMIZED_PREFILL_DTYPE"],
                    )
                    for tt_attention_mask_slice in attn_masks_unordered
                ]
            else:
                attention_mask_ = attention_mask_bool_padded * -1e3
                # Send attn masks to device
                tt_attention_mask = tt_from_torch(
                    attention_mask_,
                    dtype=ttnn.bfloat16,  # subsequent tilize op excepts bfloat16 inputs
                    device=self.mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                    mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
                )
                # Repeat attn masks for all heads
                tt_attention_mask = ttnn.repeat(
                    tt_attention_mask,
                    ttnn.Shape([1, self.config.num_attention_heads, 1, 1]),
                    memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                )
                # Tilize attn masks
                tt_attention_mask = ttnn.tilize(
                    tt_attention_mask,
                    memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                    dtype=self.model_config["ATTN_MASK_DTYPE"],
                )

            tt_input_ids = tt_from_torch(
                input_ids,
                dtype=self.model_config["INPUT_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=self.model_config["INPUT_MEMCFG"],
                mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=0),
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
            if self.model_config["l1_sharded"] == False:
                attention_mask = (attention_mask_bool_padded.transpose(0, 2) * -1e3).expand(
                    -1, self.config.num_attention_heads, -1, -1
                )
            else:
                # Reshape width to tile-size since that is required by scale_mask_softmax_in_place with causal_mask=False (in falcon_attention.py)
                attention_mask = attention_mask_bool_padded.reshape(batch_size, 1, -1, 32) * -1e3

            # Send attn masks to device
            tt_attention_mask = tt_from_torch(
                attention_mask,
                dtype=ttnn.bfloat16,  # subsequent tilize op excepts bfloat16 inputs
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )
            if not self.model_config["l1_sharded"]:
                # Tilize attn masks
                tt_attention_mask = ttnn.tilize(
                    tt_attention_mask,
                    memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                    dtype=self.model_config["ATTN_MASK_DTYPE"],
                )

            tt_input_ids = tt_from_torch(
                input_ids.transpose(0, 1),
                dtype=self.model_config["INPUT_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=self.model_config["INPUT_MEMCFG"],
                mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=1),
            )
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        return tt_input_ids, tt_attention_mask

    @abstractmethod
    def forward(
        self,
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
        device_perf_run: bool = False,
    ) -> ttnn.Tensor:
        # Convert input tokens to embeddings
        input_embeddings = ttnn.embedding(
            input_ids,
            self.embedding_weights,
            memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
        )
        input_embeddings = ttnn.unsqueeze_to_4D(input_embeddings)
        input_embeddings = ttnn.to_layout(input_embeddings, ttnn.TILE_LAYOUT)

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
            presents += (layer_output[1],)
            layer_output = layer_output[0]

            if device_perf_run and idx % 8 == 0:
                dump_device_profiler(self.mesh_device)

        # apply final norm layer
        layer_output = layernorm(
            layer_output,
            self.layernorm_eps,
            self.layernorm_gamma,
            self.layernorm_beta,
            self.model_config,
        )

        return layer_output, presents


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        mesh_device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__(
            mesh_device=mesh_device,
            state_dict=state_dict,
            base_url=base_url,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

    def forward(
        self,
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states, presents = super().forward(
            input_ids=input_ids,
            llm_mode=llm_mode,
            attention_mask=attention_mask,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )
        return hidden_states, presents
