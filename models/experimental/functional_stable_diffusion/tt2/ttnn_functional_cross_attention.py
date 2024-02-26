# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import tt_lib as ttl
from ttnn.operations.core import squeeze, unsqueeze_to_4D
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    is_tile_dim_alligned,
    round_up_to_tile_dim,
)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def pad_heads(tensor, num_heads=8, dim=-1):
    device = tensor.device()
    memory_config = ttnn.get_memory_config(tensor)

    padding_needed = not is_tile_dim_alligned(tensor.shape[dim] // num_heads)
    if padding_needed:
        tensor = ttnn_to_torch(tensor)
        unpadded_len = tensor.shape[-1] // num_heads
        padding_needed = round_up_to_tile_dim(unpadded_len) - unpadded_len
        unpadded_tensors = torch.split(tensor, tensor.shape[dim] // num_heads, dim=dim)
        padding = (
            torch.zeros((tensor.shape[-2], padding_needed))
            if dim == -1
            else torch.zeros((padding_needed, tensor.shape[-1]))
        )
        padded_tensor = torch.Tensor()
        for unpadded_tensor in unpadded_tensors:
            padded_tensor = torch.cat([padded_tensor, unpadded_tensor, padding], dim=dim)

        padded_tensor = ttnn.from_torch(padded_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        padded_tensor = ttnn.to_device(padded_tensor, device, memory_config=memory_config)
        tensor = padded_tensor

    return tensor


def concatenate_qkv(q, k, v):
    dim = -1
    device = k.device()
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

    qkv = ttnn.from_torch(qkv, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    qkv = ttnn.to_device(qkv, device, memory_config=memory_config)
    return qkv


def weight_to_bfp8(weight):
    device = weight.device()
    memory_config = ttnn.get_memory_config(weight)
    weight = ttnn_to_torch(weight)
    weight = ttnn.from_torch(weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device, memory_config=memory_config)
    return weight


class cross_attention:
    def __init__(self, device, parameters):
        self.fused_qkv = parameters.to_q.weight.shape[0] == parameters.to_k.weight.shape[0]
        for key in ["to_q", "to_k", "to_v"]:
            parameters[key].weight = pad_heads(parameters[key].weight, 8)
            assert "bias" not in parameters[key]
        parameters.to_out[0].weight = pad_heads(parameters.to_out[0].weight, 8, dim=-2)

        if self.fused_qkv:
            parameters["qkv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.qkv["weight"] = concatenate_qkv(
                parameters.to_q.weight, parameters.to_k.weight, parameters.to_v.weight
            )
        else:
            parameters["kv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.kv["weight"] = concatenate_qkv(None, parameters.to_k.weight, parameters.to_v.weight)
            parameters.to_q.weight = weight_to_bfp8(parameters.to_q.weight)

        self.parameters = parameters
        self.device = device

        scale = torch.ones((1, 1)) * 40**-0.5
        self.scale_40 = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        scale = torch.ones((1, 1)) * 80**-0.5
        self.scale_80 = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        scale = torch.ones((1, 1)) * 160**-0.5
        self.scale_160 = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.scales = {40: self.scale_40, 80: self.scale_80, 160: self.scale_160}

        attention_mask_96 = torch.ones((2, 1, 1, 96)) * -1e9
        attention_mask_96[:, :, :, :64] = 0
        attention_mask_96 = ttnn.from_torch(
            attention_mask_96, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_64 = torch.zeros((2, 1, 1, 64))
        attention_mask_64 = ttnn.from_torch(
            attention_mask_64, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_256 = torch.zeros((2, 1, 1, 256))
        attention_mask_256 = ttnn.from_torch(
            attention_mask_256, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_1024 = torch.zeros((2, 1, 1, 1024))
        attention_mask_1024 = ttnn.from_torch(
            attention_mask_1024, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_4096 = torch.zeros((2, 1, 1, 4096))
        attention_mask_4096 = ttnn.from_torch(
            attention_mask_4096, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        self.attention_masks = {
            64: attention_mask_64,
            96: attention_mask_96,
            256: attention_mask_256,
            1024: attention_mask_1024,
            4096: attention_mask_4096,
        }

        padding_shapes = [[2, 4000, 1024], [2, 928, 1536], [2, 160, 2560], [2, 32, 1280]]
        self.padded_tensors = {}
        for shape in padding_shapes:
            padding = torch.zeros(shape)
            padding = ttnn.from_torch(padding, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)
            self.padded_tensors[shape[-2]] = padding
        self.parameters.to_out[0].weight = weight_to_bfp8(self.parameters.to_out[0].weight)
        self.parameters.to_out[0].bias = weight_to_bfp8(self.parameters.to_out[0].bias)

    def get_attention_scores(self, query, t_key, attention_mask=None, scale=None, device=None):
        # t_key = ttnn.permute(key, (0, 1, 3, 2))
        attention_scores = ttnn.matmul(
            query,
            t_key,
            # core_grid=ttnn.CoreGrid(y=8, x=8)
        )
        del t_key, query
        attention_scores = ttnn.mul(attention_scores, scale)
        del scale

        if attention_mask is not None:
            attention_scores = ttnn.add(attention_scores, attention_mask)
            del attention_mask

        attention_scores = ttnn.softmax(attention_scores, dim=-1)

        return attention_scores

    def get_attention_scores_opt(self, query, t_key, head_size, attention_mask=None, device=None):
        if (
            True
        ):  # query.shape[-2] == 4096: #query.shape[-2] == 4096:#TODO: Hangs on Query shape ttnn.Shape([2, 8, 1024, 96]) and Key shape ttnn.Shape([2, 8, 96, 96])
            attention_scores = ttnn.matmul(
                query,
                t_key,
            )
            ttnn.deallocate(query)
            ttnn.deallocate(t_key)
            attention_scores = ttnn.transformer.attention_softmax(
                attention_scores, attention_mask=attention_mask, head_size=head_size
            )
        else:
            attention_scores = ttnn.matmul(
                query,
                t_key,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(query)
            ttnn.deallocate(t_key)

            attention_scores = ttnn.transformer.attention_softmax(
                attention_scores,
                attention_mask=attention_mask,
                head_size=head_size,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
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
        hidden_states_padded = False
        if len(hidden_states.shape) == 4:
            hidden_states = squeeze(hidden_states, 0)
        if encoder_hidden_states and len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = squeeze(encoder_hidden_states, 0)

        if self.fused_qkv:
            qkv_out = ttnn.matmul(
                hidden_states,
                self.parameters.qkv.weight,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=hidden_states.device().core_grid,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(hidden_states)

            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_out,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=heads,
            )
            ttnn.deallocate(qkv_out)
            attention_mask = self.attention_masks[key.shape[-1]]
        else:
            hidden_seq_len = hidden_states.shape.with_tile_padding()[-2]
            encoder_hidden_seq_len = encoder_hidden_states.shape.with_tile_padding()[-2]

            q_proj = ttnn.linear(
                hidden_states,
                self.parameters.to_q.weight,
                core_grid=hidden_states.device().core_grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(hidden_states)
            if encoder_hidden_seq_len > hidden_seq_len:
                padding_needed = encoder_hidden_seq_len - hidden_seq_len
                q_proj = ttnn.concat([q_proj, self.padded_tensors[padding_needed]], dim=1)
                hidden_states_padded = True

            kv_proj = ttnn.linear(
                encoder_hidden_states,
                self.parameters.kv.weight,
                core_grid=encoder_hidden_states.device().core_grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            # Can't deallocate, mnaybe we just move to dram if it causes issues.
            # ttnn.deallocate(encoder_hidden_states)
            if hidden_seq_len > encoder_hidden_seq_len:
                padding_needed = hidden_seq_len - encoder_hidden_seq_len
                kv_proj = ttnn.concat([kv_proj, self.padded_tensors[padding_needed]], dim=1)

            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(q_proj, kv_proj, num_heads=heads)
            ttnn.deallocate(kv_proj)
            ttnn.deallocate(q_proj)

            if key.shape[-1] > 96:
                key = key[:, :, :, :96]
            if value.shape[-2] > 96:
                value = value[:, :, :96, :]
            assert key.shape[-1] in self.attention_masks
            attention_mask = self.attention_masks[key.shape[-1]]

        if hidden_states.shape[-2] == 4096:
            query = ttnn.reallocate(query)
            key = ttnn.reallocate(key)
            value = ttnn.reallocate(value)

        attention_probs = self.get_attention_scores_opt(
            query,
            key,
            dim_head,
            attention_mask
            # query, key, attention_mask, scale=self.scales[dim_head], device=self.device
        )

        if hidden_states_padded:
            attention_probs = attention_probs[:, :, :original_seq_len, :]
        hidden_states = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        hidden_states = ttnn.transformer.concatenate_heads(
            hidden_states, memory_config=ttnn.get_memory_config(hidden_states)
        )
        if hidden_states.shape.with_tile_padding()[-1] != hidden_states.shape[-1]:
            hidden_states = hidden_states[:, :, : hidden_states.shape[-1]]

        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.to_out[0].weight,
            bias=self.parameters.to_out[0].bias,
            core_grid=hidden_states.device().core_grid,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        if len(hidden_states.shape) == 3:
            hidden_states = unsqueeze_to_4D(hidden_states)

        return hidden_states
