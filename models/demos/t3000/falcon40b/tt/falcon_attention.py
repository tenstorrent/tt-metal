# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import Optional, Tuple

import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh

from models.utility_functions import nearest_32
from models.demos.t3000.falcon40b.tt.model_utils import convert_to_layout

from models.demos.t3000.falcon40b.tt.model_utils import falcon_prefill_matmul, determine_tensor_deallocation


def generate_cos_sin_cache(
    device_mesh,
    head_dim,
    base_url,
    max_position_embeddings=2048,
    base=10000,
    model_config=None,
    tt_cache_path=None,
):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(
        max_position_embeddings,
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)

    layer_name = f"{base_url}.rotary_embedding_base_{base}_head_dim_{head_dim}_seq_len_{max_position_embeddings}"

    cos_cached_path = tt_cache_path / f"{layer_name}.cos_cached_{model_config['COS_CACHED_WEIGHTS_DTYPE'].name}"

    tt_cos_cached = ttnn.as_tensor(
        tensor=emb.cos()[None, None, :, :],
        dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=model_config["COS_CACHED_WEIGHTS_MEMCFG"],
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        cache_file_name=cos_cached_path,
    )

    sin_cached_path = tt_cache_path / f"{layer_name}.sin_cached_{model_config['SIN_CACHED_WEIGHTS_DTYPE'].name}"

    tt_sin_cached = ttnn.as_tensor(
        tensor=emb.sin()[None, None, :, :],
        dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=model_config["SIN_CACHED_WEIGHTS_MEMCFG"],
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        cache_file_name=sin_cached_path,
    )

    return tt_cos_cached, tt_sin_cached


class TtFalconRotaryEmbedding:
    """
    See FalconRotaryEmbedding from hf_modeling_falcon.py
    """

    def __init__(
        self,
        device_mesh,
        head_dim,
        base_url,
        layer_num,
        max_position_embeddings=2048,
        base=10000,
        model_config=None,
        tt_cache_path=None,
        global_cos_sin_cache=None,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.model_config = model_config
        if global_cos_sin_cache is not None:
            self.tt_cos_cached, self.tt_sin_cached = global_cos_sin_cache
        else:
            self.tt_cos_cached, self.tt_sin_cached = generate_cos_sin_cache(
                device_mesh,
                head_dim,
                f"{base_url}.{layer_num}",
                max_position_embeddings,
                base,
                model_config,
                tt_cache_path,
            )

    def __call__(
        self, layer: ttnn.experimental.tensor.Tensor, token_idx: Optional[int] = None
    ) -> ttnn.experimental.tensor.Tensor:
        seq_len = layer.get_legacy_shape()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"
        # TODO: Make rotary embedding in place
        output = ttnn.experimental.rotary_embedding(
            layer,
            self.tt_cos_cached,
            self.tt_sin_cached,
            token_idx,
            memory_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
        )

        return output


class TtFalconAttention:
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings: int = 2048,
        model_config=None,
        tt_cache_path=None,
        global_cos_sin_cache=None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_devices = device_mesh.get_num_devices()
        self.device_mesh = device_mesh
        self.state_dict = state_dict
        self.model_config = model_config
        self.num_heads_per_device = self.num_heads // device_mesh.get_num_devices()
        self.tt_cache_path = tt_cache_path
        self.max_batch_size = 32

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        layer_name = f"{base_url}.{layer_num}.self_attention"
        query_key_value_str = f"{layer_name}.query_key_value.weight"
        selfout_str = f"{layer_name}.dense.weight"

        self.query_key_value_weights = []
        self.dense_weights = []

        query_key_value_path = (
            tt_cache_path / f"{query_key_value_str}_{self.model_config['FUSED_QKV_MM_WEIGHTS_DTYPE'].name}"
        )

        self.query_key_value_weights = ttnn.as_tensor(
            tensor=self.state_dict[query_key_value_str],
            dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=-1),
            cache_file_name=query_key_value_path,
            preprocess=lambda x: torch.transpose(x.reshape(1, 1, *x.shape), -2, -1),
        )

        selfout_path = tt_cache_path / f"{selfout_str}_{self.model_config['SELFOUT_MM_WEIGHTS_DTYPE'].name}"

        self.dense_weights = ttnn.as_tensor(
            tensor=self.state_dict[selfout_str],
            dtype=self.model_config["SELFOUT_MM_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=-1),
            cache_file_name=selfout_path,
            preprocess=lambda x: torch.transpose(x.reshape(1, 1, *x.shape), -2, -1),
        )
        self.rotary_embedding = TtFalconRotaryEmbedding(
            self.device_mesh,
            self.head_dim,
            base_url,
            layer_num=layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            global_cos_sin_cache=global_cos_sin_cache,
        )
        # Fused to SM
        # self.scalar = pad_by_zero(torch.Tensor([1 / math.sqrt(self.head_dim)]), self.device)[0]
        self.scalar = 1 / math.sqrt(self.head_dim)

        # self.init_preprocessing(self.model_config["LLM_MODE"], max_position_embeddings)
        self.layer_past = None

    def initialize_kvcache(self):
        if self.layer_past is None:
            # Preloading the kvcache
            attn_cache_shape = (
                self.max_batch_size,
                self.num_kv_heads // self.num_devices,
                self.max_position_embeddings,
                self.head_dim,
            )
            kv_cache_path = self.tt_cache_path / f"empty_attn_cache{attn_cache_shape}"
            k_cache = []
            v_cache = []

            attn_cache = torch.zeros(attn_cache_shape)
            k_cache = ttnn.as_tensor(
                tensor=attn_cache,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device_mesh,
                memory_config=self.model_config["DRAM_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                cache_file_name=kv_cache_path,
            )

            v_cache = ttnn.as_tensor(
                tensor=attn_cache,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device_mesh,
                memory_config=self.model_config["DRAM_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                cache_file_name=kv_cache_path,
            )

            self.layer_past = (
                (
                    k_cache,
                    v_cache,
                ),
            )
        return self.layer_past

    def set_model_config(self, model_config):
        self.model_config = model_config

    def __call__(
        self,
        hidden_states: ttnn.experimental.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: ttnn.experimental.tensor.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[ttnn.experimental.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """
        if llm_mode == "prefill":
            return self.fwd_prefill(
                hidden_states=hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        elif llm_mode == "decode":
            return self.fwd_decode(
                hidden_states=hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            assert False

    def fwd_prefill(
        self,
        hidden_states: ttnn.experimental.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: ttnn.experimental.tensor.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[ttnn.experimental.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions
        batch = hidden_states.get_legacy_shape()[0]
        q_len = hidden_states.get_legacy_shape()[2]
        assert layer_past is not None

        # Fused query, key and value projection
        fused_query_key_value = falcon_prefill_matmul(
            hidden_states,
            self.query_key_value_weights,
            self.model_config["COMPUTE_KERNEL_CONFIG"],
            output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            grid=ttnn.CoreGrid(x=8, y=8) if q_len >= 512 else ttnn.CoreGrid(x=8, y=min(q_len // 32, 8)),
            transpose_mcast=True,
        )

        query_layer, key_layer, value_layer = ttnn.experimental.nlp_create_qkv_heads(
            fused_query_key_value,
            num_heads=self.num_heads // self.num_devices,
            num_kv_heads=self.num_kv_heads // self.num_devices,
            transpose_k_heads=False,
            memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_query_key_value.deallocate(True)

        # Rotary embeddings
        query_layer = self.rotary_embedding(query_layer)
        key_layer = self.rotary_embedding(key_layer)

        # K Cache update
        ttnn.fill_cache(
            layer_past[0],
            ttnn.experimental.typecast(
                key_layer, self.model_config["KV_CACHE_DTYPE"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            user_id,
        )
        ttnn.fill_cache(
            layer_past[1],
            ttnn.experimental.typecast(
                value_layer, self.model_config["KV_CACHE_DTYPE"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            user_id,
        )
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            is_causal=True,
            scale=self.scalar,
            program_config=self.model_config["SDPA_PROGCFG"],
        )
        # Deallocate query, key, value
        query_layer.deallocate(True)
        key_layer.deallocate(True)
        value_layer.deallocate(True)

        # Output projection
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )
        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        attn_output = falcon_prefill_matmul(
            attn_output,
            self.dense_weights,
            self.model_config["COMPUTE_KERNEL_CONFIG"],
            output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
            overwrite_subblock_h=1,
        )
        # There are references to tensors in case seq_len > 512
        # so we won't force deallocation
        should_deallocate_tensors = determine_tensor_deallocation(
            self.model_config["layernorm_params"]["slice_size"], q_len
        )
        if should_deallocate_tensors:
            hidden_states.deallocate(True)
        layer_present = layer_past if use_cache else None
        return attn_output, layer_present

    def fwd_decode(
        self,
        hidden_states: ttnn.experimental.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: ttnn.experimental.tensor.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[ttnn.experimental.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions
        batch = hidden_states.get_legacy_shape()[2]
        q_len = hidden_states.get_legacy_shape()[0]
        padded_layer_past_len = nearest_32(layer_past_len + 1)
        # We always store max_position_embeddings for kv_cache,
        # so we need separate variable to store the actual len of the kv_cache
        assert layer_past is not None
        assert layer_past_len >= 0 and layer_past_len <= self.max_position_embeddings

        # Reshard
        if self.model_config["LN_ATTN_OUTPUT_MEMCFG"] != self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"]:
            hidden_states = ttnn.sharded_to_interleaved(
                hidden_states,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )
            hidden_states = ttnn.interleaved_to_sharded(
                hidden_states,
                self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"],
            )

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = ttnn.matmul(
            hidden_states,
            self.query_key_value_weights,
            program_config=self.model_config["QKV_MM_PROGCFG"],
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        ###########
        ### TMs ###
        ###########
        if self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] != self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]:
            fused_query_key_value = ttnn.sharded_to_interleaved(
                fused_query_key_value,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )
            fused_query_key_value = ttnn.interleaved_to_sharded(
                fused_query_key_value,
                self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"],
            )
        query_layer, key_layer, value_layer = ttnn.experimental.nlp_create_qkv_heads(
            fused_query_key_value,
            num_heads=self.num_heads // self.num_devices,
            num_kv_heads=self.num_kv_heads // self.num_devices,
            transpose_k_heads=False,
            memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_query_key_value.deallocate(True)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        query_layer = self.rotary_embedding(query_layer, layer_past_len)
        key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        # Removed 1 dim from layer_past
        kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
        if kv_cache_memcfg.is_sharded():
            kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
            kv_cache_shard_shape[0] = layer_past[0].get_legacy_shape()[1] * padded_layer_past_len
            kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape
        # Update kv_cache in place
        ttnn.update_cache(
            layer_past[0],
            key_layer,
            layer_past_len,
        )
        key_layer.deallocate(True)

        # key and value layers will have kv_seq_len padded to nearest 32
        key_layer = ttnn.slice(
            layer_past[0],
            [0, 0, 0, 0],
            [
                batch - 1,
                self.num_kv_heads // self.num_devices - 1,
                padded_layer_past_len - 1,
                self.head_dim - 1,
            ],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        key_layer = ttnn.interleaved_to_sharded(
            key_layer,
            kv_cache_memcfg,
        )
        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        # TODO: Sharded transpose could be in place???
        key_layer_transposed = ttnn.transpose(
            key_layer,
            -2,
            -1,
            memory_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
        )
        key_layer.deallocate(True)

        attn_weights = ttnn.experimental.group_attn_matmul(
            query_layer,
            key_layer_transposed,
            compute_with_storage_grid_size=self.device_mesh.get_devices()[
                0
            ].compute_with_storage_grid_size(),  # Change this
            memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
        )
        query_layer.deallocate(True)
        key_layer_transposed.deallocate(True)

        ###############
        ### SOFTMAX ###
        ###############
        softmax_progcfg = self.model_config["SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = padded_layer_past_len // 32

        attn_weights = ttnn.scale_mask_softmax_in_place(
            attn_weights,
            self.scalar,
            attention_mask,
            program_config=self.model_config["SOFTMAX_PROGCFG"],
            is_causal_mask=True,
        )
        ######################
        ### V CACHE UPDATE ###
        ######################

        # Update kv_cache in place
        ttnn.update_cache(
            layer_past[1],
            value_layer,
            layer_past_len,
        )
        value_layer.deallocate(True)

        value_layer = ttnn.slice(
            layer_past[1],
            [0, 0, 0, 0],
            [
                batch - 1,
                self.num_kv_heads // self.num_devices - 1,
                padded_layer_past_len - 1,
                self.head_dim - 1,
            ],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        value_layer = ttnn.interleaved_to_sharded(
            value_layer,
            kv_cache_memcfg,
        )

        layer_present = layer_past if use_cache else None
        ########################
        ### POST-SOFTMAX MM ###
        ########################
        attn_output = ttnn.experimental.group_attn_matmul(
            attn_weights,
            value_layer,
            compute_with_storage_grid_size=self.device_mesh.get_devices()[0].compute_with_storage_grid_size(),
            memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
        )
        attn_weights.deallocate(True)
        value_layer.deallocate(True)
        #########################
        ### ATTENTION SELFOUT ###
        #########################
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )
        attn_output = ttnn.sharded_to_interleaved(
            attn_output,
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        attn_output = ttnn.interleaved_to_sharded(
            attn_output,
            self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"],
        )
        attn_output = ttnn.matmul(
            attn_output,
            self.dense_weights,
            program_config=self.model_config["SELFOUT_MM_PROGCFG"],
            memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        return attn_output, layer_present
