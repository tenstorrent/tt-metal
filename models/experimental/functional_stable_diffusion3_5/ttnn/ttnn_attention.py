# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_rms_norm import ttnn_RMSNorm
import torch.nn.functional as F

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig


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
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        else:
            self.norm_q = ttnn_RMSNorm(dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_q)
            self.norm_k = ttnn_RMSNorm(dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_k)
        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q = ttnn_RMSNorm(
                    dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_added_q
                )
                self.norm_added_k = ttnn_RMSNorm(
                    dim=dim_head, eps=eps, elementwise_affine=True, parameters=parameters.norm_added_k
                )
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

    def __call__(self, ttnn_Attention, hidden_states, encoder_hidden_states, attention_mask, device):
        print(hidden_states.memory_config())
        batch_size = hidden_states.shape[0]
        residual = hidden_states
        query = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_q_weight,
            bias=ttnn_Attention.to_q_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(query.memory_config())
        key = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_k_weight,
            bias=ttnn_Attention.to_k_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(key.memory_config())
        value = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_v_weight,
            bias=ttnn_Attention.to_v_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(value.memory_config())
        inner_dim = key.shape[-1]
        head_dim = inner_dim // ttnn_Attention.heads
        query = ttnn.reshape(query, (batch_size, query.shape[1], ttnn_Attention.heads, head_dim))
        print(query.memory_config())
        query = ttnn.permute(query, (0, 2, 1, 3))
        print(query.memory_config())
        key = ttnn.reshape(key, (batch_size, key.shape[1], ttnn_Attention.heads, head_dim))
        print(key.memory_config())
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.reshape(value, (batch_size, value.shape[1], ttnn_Attention.heads, head_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))
        print(value.memory_config())
        if ttnn_Attention.norm_q is not None:
            query = ttnn_Attention.norm_q(query, device)
            print(query.memory_config())
        if ttnn_Attention.norm_k is not None:
            key = ttnn_Attention.norm_k(key, device)
            print(key.memory_config())
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = ttnn.linear(
                encoder_hidden_states,
                ttnn_Attention.add_q_proj_weight,
                bias=ttnn_Attention.add_q_proj_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            print(encoder_hidden_states_query_proj.memory_config())
            encoder_hidden_states_key_proj = ttnn.linear(
                encoder_hidden_states,
                ttnn_Attention.add_k_proj_weight,
                bias=ttnn_Attention.add_k_proj_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            print(encoder_hidden_states_key_proj.memory_config())
            encoder_hidden_states_value_proj = ttnn.linear(
                encoder_hidden_states,
                ttnn_Attention.add_v_proj_weight,
                bias=ttnn_Attention.add_v_proj_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            print(encoder_hidden_states_value_proj.memory_config())
            encoder_hidden_states_query_proj = ttnn.reshape(
                encoder_hidden_states_query_proj,
                (batch_size, encoder_hidden_states_query_proj.shape[1], ttnn_Attention.heads, head_dim),
            )
            print(encoder_hidden_states_query_proj.memory_config())
            encoder_hidden_states_query_proj = ttnn.permute(encoder_hidden_states_query_proj, (0, 2, 1, 3))
            encoder_hidden_states_key_proj = ttnn.reshape(
                encoder_hidden_states_key_proj,
                (batch_size, encoder_hidden_states_key_proj.shape[1], ttnn_Attention.heads, head_dim),
            )
            print(encoder_hidden_states_key_proj.memory_config())
            encoder_hidden_states_key_proj = ttnn.permute(encoder_hidden_states_key_proj, (0, 2, 1, 3))
            encoder_hidden_states_value_proj = ttnn.reshape(
                encoder_hidden_states_value_proj,
                (batch_size, encoder_hidden_states_value_proj.shape[1], ttnn_Attention.heads, head_dim),
            )
            print(encoder_hidden_states_key_proj.memory_config())
            encoder_hidden_states_value_proj = ttnn.permute(encoder_hidden_states_value_proj, (0, 2, 1, 3))
            if ttnn_Attention.norm_added_q is not None:
                encoder_hidden_states_query_proj = ttnn_Attention.norm_added_q(
                    encoder_hidden_states_query_proj, device=device
                )
                print(encoder_hidden_states_query_proj.memory_config())
            if ttnn_Attention.norm_added_k is not None:
                encoder_hidden_states_key_proj = ttnn_Attention.norm_added_k(
                    encoder_hidden_states_key_proj, device=device
                )
                print(encoder_hidden_states_key_proj.memory_config())
            query = ttnn.concat([query, encoder_hidden_states_query_proj], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            key = ttnn.concat([key, encoder_hidden_states_key_proj], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            value = ttnn.concat([value, encoder_hidden_states_value_proj], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            query = ttnn.to_memory_config(query, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            key = ttnn.to_memory_config(key, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            value = ttnn.to_memory_config(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # if encoder_hidden_states is None:
        #     q_size = 32
        # elif encoder_hidden_states is not None and encoder_hidden_states.shape[1] == 333:
        #     q_size = 4448
        # elif encoder_hidden_states is not None and encoder_hidden_states.shape[1] == 154:
        #     q_size = 1184
        # program_config = ttnn.SDPAProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     q_chunk_size=q_size,
        #     k_chunk_size=q_size,
        #     exp_approx_mode=False,
        # )
        # compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        #     math_fidelity=ttnn.MathFidelity.HiFi4,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=False,
        #     packer_l1_acc=False,
        # )
        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
        )

        #     compute_kernel_config=compute_kernel_config,
        #     program_config=program_config,
        # )
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(hidden_states.memory_config())
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1, 3))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, -1, ttnn_Attention.heads * head_dim))
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1], :],
                hidden_states[:, residual.shape[1] :, :],
            )
            print(hidden_states.memory_config(), encoder_hidden_states.memory_config())
            if not ttnn_Attention.context_pre_only:
                encoder_hidden_states = ttnn.linear(
                    encoder_hidden_states,
                    ttnn_Attention.to_add_out_weight,
                    bias=ttnn_Attention.to_add_out_bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                print(encoder_hidden_states.memory_config())
        hidden_states = ttnn.linear(
            hidden_states,
            ttnn_Attention.to_out_weight,
            bias=ttnn_Attention.to_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(hidden_states.memory_config())
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
