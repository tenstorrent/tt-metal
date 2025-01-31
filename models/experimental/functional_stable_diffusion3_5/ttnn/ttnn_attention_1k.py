# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_rms_norm import ttnn_RMSNorm
import torch.nn.functional as F


class ttnn_Attention:
    def __init__(
        self,
        parameters,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["ttnn_JointAttnProcessor2_0"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
    ):
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.eps = eps
        if qk_norm is not None:
            self.norm_q = ttnn_RMSNorm(dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_q)
            self.norm_k = ttnn_RMSNorm(dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_k)
            self.param_norm_q = parameters.norm_q
            self.param_norm_k = parameters.norm_k
        else:
            self.norm_q = None
            self.norm_k = None
        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q = ttnn_RMSNorm(
                    dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_added_q
                )
                self.norm_added_k = ttnn_RMSNorm(
                    dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_added_k
                )
                self.param_norm_added_q = parameters.norm_added_q
                self.param_norm_added_k = parameters.norm_added_k
        else:
            self.norm_added_q = None
            self.norm_added_k = None
        self.to_q_weight = parameters.to_q.weight
        self.to_q_bias = parameters.to_q.bias
        self.to_k_weight = parameters.to_k.weight
        self.to_k_bias = parameters.to_k.bias
        self.to_v_weight = parameters.to_v.weight
        self.to_v_bias = parameters.to_v.bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj_weight = parameters.add_k_proj.weight
            if added_proj_bias:
                self.add_k_proj_bias = parameters.add_k_proj.bias
            self.add_v_proj_weight = parameters.add_v_proj.weight
            if added_proj_bias:
                self.add_v_proj_bias = parameters.add_v_proj.bias
            if self.context_pre_only is not None:
                self.add_q_proj_weight = parameters.add_q_proj.weight
                if added_proj_bias:
                    self.add_q_proj_bias = parameters.add_q_proj.bias
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out_weight = parameters.to_add_out.weight
            self.to_add_out_bias = parameters.to_add_out.bias
        if not self.pre_only:
            self.to_out_weight = parameters.to_out[0].weight
            self.to_out_bias = parameters.to_out[0].bias
        self.processor = processor

    def __call__(self, hidden_states, encoder_hidden_states=None, attention_mask=None, device=None):
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = self.processor(
                self, hidden_states, encoder_hidden_states, attention_mask, device
            )
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.processor(self, hidden_states, encoder_hidden_states, attention_mask, device)
            return hidden_states


class ttnn_JointAttnProcessor2_0:
    def __init__(self):
        pass

    def __call__(self, ttnn_Attention, hidden_states_i, encoder_hidden_states_i, attention_mask, device=None):
        batch_size = hidden_states_i.shape[0]
        # residual = hidden_states
        residual_shape = hidden_states_i.shape

        ## Main Lineaer
        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        hidden_states = ttnn.to_memory_config(
            hidden_states_i,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states_i.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(hidden_states_i)
        hidden_states = ttnn.reallocate(hidden_states)
        ####

        ## Proj Linear
        encoder_hidden_states_exist = 0
        if encoder_hidden_states_i is not None:
            encoder_hidden_states_exist = 1
            if encoder_hidden_states_i.shape[-2] < 512:
                mm_a_x_proj = 8
                mm_a_y_proj = 6
                mm_a_x_strategy_proj = ttnn.ShardStrategy.WIDTH
                mm_a_x_memory_config_proj = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            encoder_hidden_states = ttnn.to_memory_config(
                encoder_hidden_states_i,
                memory_config=ttnn.create_sharded_memory_config(
                    encoder_hidden_states_i.shape,
                    core_grid=ttnn.CoreGrid(y=mm_a_y_proj, x=mm_a_x_proj),
                    strategy=mm_a_x_strategy_proj,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(encoder_hidden_states_i)
            encoder_hidden_states = ttnn.reallocate(encoder_hidden_states)

        #####
        query = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_q_weight,
            bias=ttnn_Attention.to_q_bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )
        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        query = ttnn.experimental.nlp_create_qkv_heads_sd35(query, memory_config=ttnn.L1_MEMORY_CONFIG)[0]
        if ttnn_Attention.norm_q is not None:
            query = ttnn.rms_norm(query, epsilon=ttnn_Attention.eps, weight=ttnn_Attention.param_norm_q.weight)
        if encoder_hidden_states_exist:
            encoder_hidden_states_query_proj = ttnn.linear(
                encoder_hidden_states,
                ttnn_Attention.add_q_proj_weight,
                bias=ttnn_Attention.add_q_proj_bias,
                memory_config=mm_a_x_memory_config_proj,
                core_grid=ttnn.CoreGrid(y=mm_a_y_proj, x=mm_a_x_proj),
            )
            ## Split Head
            encoder_hidden_states_query_proj = ttnn.to_memory_config(
                encoder_hidden_states_query_proj, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
            )
            encoder_hidden_states_query_proj = ttnn.experimental.nlp_create_qkv_heads_sd35(
                encoder_hidden_states_query_proj, memory_config=ttnn.L1_MEMORY_CONFIG
            )[0]
            #
            if ttnn_Attention.norm_added_q is not None:
                encoder_hidden_states_query_proj = ttnn.rms_norm(
                    encoder_hidden_states_query_proj,
                    epsilon=ttnn_Attention.eps,
                    weight=ttnn_Attention.param_norm_added_q.weight,
                )
            query = ttnn.concat([query, encoder_hidden_states_query_proj], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(encoder_hidden_states_query_proj)
        else:
            query = ttnn.to_memory_config(query, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ######
        key = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_k_weight,
            bias=ttnn_Attention.to_k_bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        key = ttnn.experimental.nlp_create_qkv_heads_sd35(key, memory_config=ttnn.L1_MEMORY_CONFIG)[0]
        if ttnn_Attention.norm_k is not None:
            key = ttnn.rms_norm(key, epsilon=ttnn_Attention.eps, weight=ttnn_Attention.param_norm_k.weight)
        if encoder_hidden_states_exist:
            encoder_hidden_states_key_proj = ttnn.linear(
                encoder_hidden_states,
                ttnn_Attention.add_k_proj_weight,
                bias=ttnn_Attention.add_k_proj_bias,
                memory_config=mm_a_x_memory_config_proj,
                core_grid=ttnn.CoreGrid(y=mm_a_y_proj, x=mm_a_x_proj),
            )
            encoder_hidden_states_key_proj = ttnn.to_memory_config(
                encoder_hidden_states_key_proj, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
            )
            encoder_hidden_states_key_proj = ttnn.experimental.nlp_create_qkv_heads_sd35(
                encoder_hidden_states_key_proj, memory_config=ttnn.L1_MEMORY_CONFIG
            )[0]
            #
            if ttnn_Attention.norm_added_k is not None:
                encoder_hidden_states_key_proj = ttnn.rms_norm(
                    encoder_hidden_states_key_proj,
                    epsilon=ttnn_Attention.eps,
                    weight=ttnn_Attention.param_norm_added_k.weight,
                )
            key = ttnn.concat([key, encoder_hidden_states_key_proj], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(encoder_hidden_states_key_proj)
        else:
            key = ttnn.to_memory_config(key, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        #####
        value = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_v_weight,
            bias=ttnn_Attention.to_v_bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        value = ttnn.experimental.nlp_create_qkv_heads_sd35(value, memory_config=ttnn.L1_MEMORY_CONFIG)[0]
        ttnn.deallocate(hidden_states)
        # value = ttnn.reallocate(value)
        if encoder_hidden_states_exist:
            encoder_hidden_states_value_proj = ttnn.linear(
                encoder_hidden_states,
                ttnn_Attention.add_v_proj_weight,
                bias=ttnn_Attention.add_v_proj_bias,
                memory_config=mm_a_x_memory_config_proj,
                core_grid=ttnn.CoreGrid(y=mm_a_y_proj, x=mm_a_x_proj),
            )
            encoder_hidden_states_value_proj = ttnn.to_memory_config(
                encoder_hidden_states_value_proj, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
            )
            ttnn.deallocate(encoder_hidden_states)
            encoder_hidden_states_value_proj = ttnn.experimental.nlp_create_qkv_heads_sd35(
                encoder_hidden_states_value_proj, memory_config=ttnn.L1_MEMORY_CONFIG
            )[0]
            value = ttnn.concat([value, encoder_hidden_states_value_proj], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(encoder_hidden_states_value_proj)
        else:
            value = ttnn.to_memory_config(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ## SDPA
        hidden_states_combined = ttnn.transformer.scaled_dot_product_attention(query, key, value, is_causal=False)
        hidden_states_combined = ttnn.to_memory_config(hidden_states_combined, memory_config=ttnn.L1_MEMORY_CONFIG)

        ## Concat Heads
        hidden_states_combined = ttnn.experimental.nlp_concat_heads(
            hidden_states_combined,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        )

        if encoder_hidden_states_exist:
            # hidden_states, encoder_hidden_states = (
            #     hidden_states[:, :, : residual.shape[-2], :],
            #     hidden_states[:, :, residual.shape[-2] :, :],
            # )
            # residual_shape = residual.shape
            dim_hidden = residual_shape[-1]
            seq_len_main = residual_shape[-2]
            seq_len_combined = hidden_states_combined.shape[-2]

            encoder_hidden_states = ttnn.slice(
                hidden_states_combined, [0, 0, seq_len_main, 0], [batch_size, 1, seq_len_combined, dim_hidden]
            )
            hidden_states = ttnn.slice(hidden_states_combined, [0, 0, 0, 0], [batch_size, 1, seq_len_main, dim_hidden])
            ttnn.deallocate(hidden_states_combined)
            # hidden_states = ttnn.reallocate(hidden_states)

            if not ttnn_Attention.context_pre_only:
                if encoder_hidden_states.shape[-2] < 512:
                    mm_a_y = 6
                    mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
                    mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
                encoder_hidden_states = ttnn.to_memory_config(
                    encoder_hidden_states,
                    memory_config=ttnn.create_sharded_memory_config(
                        encoder_hidden_states.shape,
                        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                        strategy=mm_a_x_strategy,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat8_b,
                )
                encoder_hidden_states = ttnn.linear(
                    encoder_hidden_states,
                    ttnn_Attention.to_add_out_weight,
                    bias=ttnn_Attention.to_add_out_bias,
                    memory_config=mm_a_x_memory_config,
                    core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                )

        else:
            hidden_states = hidden_states_combined

        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        hidden_states_last = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(hidden_states)
        hidden_states_last = ttnn.reallocate(hidden_states_last)
        if encoder_hidden_states_exist:
            encoder_hidden_states = ttnn.reallocate(encoder_hidden_states)

        hidden_states_last = ttnn.linear(
            hidden_states_last,
            ttnn_Attention.to_out_weight,
            bias=ttnn_Attention.to_out_bias,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )
        if encoder_hidden_states_exist:
            return hidden_states_last, encoder_hidden_states
        else:
            return hidden_states_last
