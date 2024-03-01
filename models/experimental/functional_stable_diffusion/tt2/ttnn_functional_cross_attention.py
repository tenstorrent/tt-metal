# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import tt_lib as ttl
from ttnn.operations.core import squeeze, unsqueeze_to_4D
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import is_tile_dim_alligned


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def concatenate_qkv(q, k, v):
    dim = -1
    device = k.device
    memory_config = ttnn.get_memory_config(k)

    if q is not None:
        q = ttnn_to_torch(q)
        assert is_tile_dim_alligned(q.shape[dim])

    k = ttnn_to_torch(k)
    v = ttnn_to_torch(v)

    assert is_tile_dim_alligned(k.shape[dim])
    assert is_tile_dim_alligned(v.shape[dim])

    if q is not None:
        qkv = torch.cat([q, k, v], dim=dim)
    else:
        qkv = torch.cat([k, v], dim=dim)
    qkv = ttnn.from_torch(qkv, ttnn.bfloat16)
    qkv = ttnn.to_layout(qkv, layout=ttnn.TILE_LAYOUT)
    qkv = ttnn.to_device(qkv, device, memory_config=memory_config)
    return qkv


class cross_attention:
    def __init__(self, device, parameters):
        self.fused_qkv = parameters.to_q.weight.shape[0] == parameters.to_k.weight.shape[0]
        if self.fused_qkv:
            parameters["qkv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.qkv["weight"] = concatenate_qkv(
                parameters.to_q.weight, parameters.to_k.weight, parameters.to_v.weight
            )

            for key in ["to_q", "to_k", "to_v"]:
                assert "bias" not in parameters[key]
                # del parameters[key]
        else:
            parameters["kv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.kv["weight"] = concatenate_qkv(None, parameters.to_k.weight, parameters.to_v.weight)

        self.device = device
        self.parameters = parameters

        scale = torch.ones((1, 1)) * 40**-0.5
        self.scale_40 = ttnn.from_torch(scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        scale = torch.ones((1, 1)) * 80**-0.5
        self.scale_80 = ttnn.from_torch(scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        scale = torch.ones((1, 1)) * 160**-0.5
        self.scale_160 = ttnn.from_torch(scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.scales = {40: self.scale_40, 80: self.scale_80, 160: self.scale_160}

        attention_mask_96 = torch.ones((1, 1, 1, 96)) * -1e9
        attention_mask_96[:, :, :, :77] = 0
        attention_mask_96 = ttnn.from_torch(
            attention_mask_96, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_256 = torch.ones((1, 1, 1, 256)) * -1e9
        attention_mask_256[:, :, :, :77] = 0
        attention_mask_256 = ttnn.from_torch(
            attention_mask_256, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_1024 = torch.ones((1, 1, 1, 1024)) * -1e9
        attention_mask_1024[:, :, :, :77] = 0
        attention_mask_1024 = ttnn.from_torch(
            attention_mask_1024, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_4096 = torch.ones((1, 1, 1, 4096)) * -1e9
        attention_mask_4096[:, :, :, :77] = 0
        attention_mask_4096 = ttnn.from_torch(
            attention_mask_4096, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # if key.shape[-1] == 512:
        #     attention_mask[:, :, :, 256 : 256 + 77] = 0

        self.attention_masks = {
            96: attention_mask_96,
            256: attention_mask_256,
            1024: attention_mask_1024,
            4096: attention_mask_4096,
        }

    def get_attention_scores(self, query, t_key, attention_mask=None, scale=None, device=None):
        # t_key = ttnn.permute(key, (0, 1, 3, 2))
        attention_scores = ttnn.matmul(query, t_key, core_grid=ttnn.CoreGrid(y=8, x=8))
        del t_key, query
        attention_scores = ttnn.mul(attention_scores, scale)
        del scale

        if attention_mask is not None:
            attention_scores = ttnn.add(attention_scores, attention_mask)
            del attention_mask

        attention_scores = ttnn.softmax(attention_scores, dim=-1)

        return attention_scores

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        query_dim: int = None,
        cross_attention_dim=None,
        heads: int = 8,
        dim_head: int = 64,
        attention_mask=None,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_kwargs={},
    ):
        assert dim_head in self.scales
        original_seq_len = hidden_states.shape[-2]
        if len(hidden_states.shape) == 4:
            hidden_states = squeeze(hidden_states, 0)
        if encoder_hidden_states and len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = squeeze(encoder_hidden_states, 0)

        if self.fused_qkv:
            qkv_out = ttnn.matmul(
                hidden_states,
                self.parameters.qkv.weight,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            ttnn.deallocate(hidden_states)
            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_out,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=heads,
            )
            ttnn.deallocate(qkv_out)
            # if query.shape[2] == 4096:
            #     query = ttnn.reallocate(query)
            #     key = ttnn.reallocate(key)
            #     value = ttnn.reallocate(value)
        else:
            hidden_seq_len = hidden_states.shape.with_tile_padding()[-2]
            encoder_hidden_seq_len = encoder_hidden_states.shape.with_tile_padding()[-2]

            q_proj = ttnn.linear(
                hidden_states,
                self.parameters.to_q.weight,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            del hidden_states
            if encoder_hidden_seq_len > hidden_seq_len:
                padding_needed = encoder_hidden_seq_len - hidden_seq_len
                q_proj = ttnn.pad(q_proj, ((0, 0), (0, padding_needed), (0, 0)), 0)
            kv_proj = ttnn.linear(
                encoder_hidden_states,
                self.parameters.kv.weight,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            del encoder_hidden_states
            if hidden_seq_len > encoder_hidden_seq_len:
                padding_needed = hidden_seq_len - encoder_hidden_seq_len
                kv_proj = ttnn.pad(kv_proj, ((0, 0), (0, padding_needed), (0, 0)), 0)
            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(q_proj, kv_proj, num_heads=heads)
            del kv_proj
            del q_proj

            assert key.shape[-1] in self.attention_masks
            attention_mask = self.attention_masks[key.shape[-1]]

        attention_probs = self.get_attention_scores(
            query, key, attention_mask, scale=self.scales[dim_head], device=self.device
        )

        hidden_states = ttnn.matmul(attention_probs, value)

        if hidden_states.shape[-2] != original_seq_len:
            hidden_states = hidden_states[:, :, :original_seq_len, :]
        hidden_states = ttnn.transformer.concatenate_heads(
            hidden_states, memory_config=ttnn.get_memory_config(hidden_states)
        )

        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.to_out[0].weight,
            bias=self.parameters.to_out[0].bias,
            memory_config=ttnn.get_memory_config(hidden_states),
        )

        if len(hidden_states.shape) == 3:
            hidden_states = unsqueeze_to_4D(hidden_states)

        return ttnn.unsqueeze_to_4D(hidden_states)  # TODO: This wouldn't be needed if the input was 3D ...
