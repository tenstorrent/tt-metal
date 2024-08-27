# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn

from models.utility_functions import (
    nearest_32,
    is_wormhole_b0,
)
from .falcon_rotary_embedding import TtFalconRotaryEmbedding


class TtFalconAttention:
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        model_config=None,
        parameters=None,
        core_grid=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.parameters = parameters
        self.model_config = model_config

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        self.query_key_value_weights = self.parameters.query_key_value.weight
        self.dense_weights = self.parameters.dense.weight

        self.rotary_embedding = TtFalconRotaryEmbedding(
            parameters=parameters.rotary_emb,
            model_config=model_config,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.scalar = 1 / math.sqrt(self.head_dim)
        self.core_grid = core_grid

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        alibi: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """

        assert not output_attentions

        if llm_mode == "prefill":
            batch = hidden_states.shape[0]
            q_len = hidden_states.shape[2]
            assert layer_past is not None
        elif llm_mode == "decode":
            batch = hidden_states.shape[2]
            q_len = hidden_states.shape[0]
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        if isinstance(layer_past, tuple):
            layer_past = list(layer_past)

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = ttnn.matmul(
            input_tensor_a=hidden_states,
            input_tensor_b=self.query_key_value_weights,
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            core_grid=self.core_grid,
        )
        batch_size, _, sequence_size, fused_query_key_value_width = fused_query_key_value.shape
        fused_query_key_value = ttnn.reshape(
            fused_query_key_value, (batch_size, sequence_size, fused_query_key_value_width)
        )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_query_key_value,
            num_heads=self.num_heads,
            num_kv_heads=1,
            transpose_key=False,
            memory_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        ttnn.deallocate(fused_query_key_value)

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        if llm_mode == "prefill":
            query_layer = self.rotary_embedding(query_layer)
            key_layer = self.rotary_embedding(key_layer)
        elif llm_mode == "decode":
            query_layer = self.rotary_embedding(query_layer, layer_past_len)
            key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            ttnn.kv_cache.fill_cache_for_user_(layer_past[0], key_layer, user_id)

        elif llm_mode == "decode":
            ttnn.kv_cache.update_cache_for_token_(layer_past[0], key_layer, layer_past_len)
            # key and value layers will have kv_seq_len padded to nearest 32
            # memory_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"] TODO: add memory_config to __getitem__
            key_layer = layer_past[0][:, :, : nearest_32(layer_past_len + 1), : self.head_dim]

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = ttnn.permute(
            key_layer,
            (0, 1, 3, 2),
            # memory_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"], TODO(cfjchu)
        )
        ttnn.deallocate(key_layer, force=False)

        if llm_mode == "prefill":
            attn_weights = ttnn.matmul(
                input_tensor_a=query_layer,
                input_tensor_b=key_layer_transposed,
                memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                core_grid=self.core_grid,
            )

        elif llm_mode == "decode":
            # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
            if is_wormhole_b0():
                attn_weights = ttnn.experimental.attn_matmul(
                    query_layer,
                    key_layer_transposed,
                    compute_with_storage_grid_size=ttnn.CoreCoord(self.core_grid.x, self.core_grid.y),
                    memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
            else:
                attn_weights = ttnn.experimental.group_attn_matmul(
                    query_layer,
                    key_layer_transposed,
                    compute_with_storage_grid_size=ttnn.CoreCoord(self.core_grid.x, self.core_grid.y),
                    memory_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer_transposed)

        attn_weights = ttnn.mul(
            attn_weights, self.scalar, memory_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"]
        )

        if attention_mask is not None:
            attn_weights = ttnn.add(
                attn_weights,
                attention_mask,
                memory_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
            )

        ###############
        ### SOFTMAX ###
        ###############
        # TODO: Replace with scaled_softmax_attention_mask from BERT
        # attn_weights = ttnn.mul(attn_weights, 1 / self.temperature)
        # attn_weights = ttnn.transformer.attention_softmax(attn_weights)
        attn_weights = ttnn.softmax(attn_weights, dim=-1)  # TODO(jchu): change to in-place

        ######################
        ### V CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            ttnn.kv_cache.fill_cache_for_user_(layer_past[1], value_layer, user_id)

        elif llm_mode == "decode":
            ttnn.kv_cache.update_cache_for_token_(layer_past[1], value_layer, layer_past_len)
            # memory_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG" TODO: add memory_config to __getitem__
            value_layer = layer_past[1][:, :, : nearest_32(layer_past_len + 1), : self.head_dim]

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        if llm_mode == "prefill":
            attn_output = ttnn.matmul(
                attn_weights,
                value_layer,
                memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            )

        elif llm_mode == "decode":
            # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
            if is_wormhole_b0():
                attn_output = ttnn.experimental.attn_matmul(
                    attn_weights,
                    value_layer,
                    compute_with_storage_grid_size=ttnn.CoreCoord(self.core_grid.x, self.core_grid.y),
                    memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
            else:
                attn_output = ttnn.experimental.group_attn_matmul(
                    attn_weights,
                    value_layer,
                    compute_with_storage_grid_size=ttnn.CoreCoord(self.core_grid.x, self.core_grid.y),
                    memory_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
        ttnn.deallocate(attn_weights)
        ttnn.deallocate(value_layer, force=False)

        #########################
        ### ATTENTION SELFOUT ###
        #########################
        attn_output = ttnn.transformer.concatenate_heads(
            attn_output,
            memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )

        attn_output = ttnn.linear(
            attn_output,
            self.dense_weights,
            memory_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            core_grid=self.core_grid,
        )

        return attn_output, layer_present
