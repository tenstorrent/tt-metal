# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import tt_lib as ttl
from tt_lib.fallback_ops import fallback_ops
from tests.tt_eager.python_api_testing.fused_ops.softmax import softmax as TtSoftmax

from models.stable_diffusion.sd_utils import make_linear


class TtCrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        processor: Optional["AttnProcessor"] = None,
        device=None,
        host=None,
        state_dict=None,
        base_address="",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.device = device
        self.host = host
        self.out_mem_config_l1 = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1)

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.torch_group_norm_weight = nn.Parameter(state_dict[f"{base_address}.torch_group_norm.weight"])
            self.torch_group_norm_bias = nn.Parameter(state_dict[f"{base_address}.torch_group_norm.bias"])
            self.torch_group_norm = fallback_ops.GroupNorm(self.torch_group_norm_weight, self.torch_group_norm_bias, num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.torch_group_norm = None


        qweights = state_dict[f"{base_address}.to_q.weight"]
        qbias = state_dict[f"{base_address}.to_q.bias"] if bias else None
        self.to_q = make_linear(in_features=query_dim, out_features=inner_dim, weights=qweights, bias=qbias, device=self.device)


        kweights = state_dict[f"{base_address}.to_k.weight"]
        kbias = state_dict[f"{base_address}.to_k.bias"] if bias else None
        self.to_k = make_linear(in_features=cross_attention_dim, out_features=inner_dim, weights=kweights, bias=kbias, device=self.device)



        vweights = state_dict[f"{base_address}.to_v.weight"]
        vbias = state_dict[f"{base_address}.to_v.bias"] if bias else None
        self.to_v = make_linear(in_features=cross_attention_dim, out_features=inner_dim, weights=vweights, bias=vbias, device=self.device)


        if self.added_kv_proj_dim is not None:
            add_k_proj_weights = state_dict[f"{base_address}.add_k_proj.weight"]
            add_k_proj_bias = state_dict[f"{base_address}.add_k_proj.bias"] if bias else None
            self.add_k_proj = make_linear(in_features=added_kv_proj_dim, out_features=cross_attention_dim, weights=add_k_proj_weights, bias=add_k_proj_bias, device=self.device)

            add_v_proj_weights = state_dict[f"{base_address}.add_v_proj.weight"]
            add_v_proj_bias = state_dict[f"{base_address}.add_v_proj.bias"] if bias else None
            self.add_v_proj = make_linear(in_features=added_kv_proj_dim, out_features=cross_attention_dim, weights=add_v_proj_weights, bias=add_v_proj_bias, device=self.device)


        to_out0_weight = state_dict[f"{base_address}.to_out.0.weight"]
        to_out0_bias = state_dict[f"{base_address}.to_out.0.bias"]
        self.to_out = make_linear(in_features=inner_dim, out_features=query_dim, weights=to_out0_weight, bias=to_out0_bias, device=self.device)


    def set_attention_slice(self, slice_size):
        assert False, "Not Implemented"

    def set_processor(self, processor: "AttnProcessor"):
        self.processor = processor

    def forward(self, hidden_states: ttl.tensor.Tensor, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs) -> ttl.tensor.Tensor:
        # The `CrossAttention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return CrossAttnProcessor(self, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,)

    def batch_to_head_dim(self, tensor: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        head_size = self.heads
        _, batch_size, seq_len, dim = tensor.shape()
        tensor = fallback_ops.reshape(tensor, batch_size // head_size, head_size, seq_len, dim)
        tensor = ttl.tensor.permute(tensor, (0, 2, 1, 3))
        tensor = fallback_ops.reshape(tensor, 1, batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        head_size = self.heads
        _, batch_size, seq_len, dim = tensor.shape()
        tensor = fallback_ops.reshape(tensor, batch_size, seq_len, head_size, dim // head_size)
        tensor = ttl.tensor.permute(tensor, (0, 2, 1, 3))
        tensor = fallback_ops.reshape(tensor, 1, batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query: ttl.tensor.Tensor, key: ttl.tensor.Tensor, attention_mask=None) -> ttl.tensor.Tensor:
        t_key = ttl.tensor.transpose(key)

        temp = ttl.tensor.bmm(query, t_key)
        # Aaron: TODO: intentionally keeping this here!
        # scale_tensor = ttl.tensor.fill_rm(temp.shape()[0],
        #                                 temp.shape()[1],
        #                                 temp.shape()[2],
        #                                 temp.shape()[3],
        #                                 0,
        #                                 0,
        #                                 temp,
        #                                 self.scale,
        #                                 self.scale)

        scale_tensor = fallback_ops.full(temp.shape(), self.scale)
        attention_scores = ttl.tensor.mul(scale_tensor, temp)

        if attention_mask is not None:
            attention_scores = ttl.tensor.add(attention_scores, attention_mask)

        attention_probs = TtSoftmax(attention_scores)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            assert False, "Attention Mask has always been None, This is not implemented!"

        return attention_mask


def CrossAttnProcessor(attn: TtCrossAttention, hidden_states: ttl.tensor.Tensor, encoder_hidden_states=None, attention_mask=None) -> ttl.tensor.Tensor:
    out_mem_config_l1 = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1)

    _, batch_size, sequence_length, _ = hidden_states.shape()
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)
    query = attn.to_q(hidden_states)
    query = attn.head_to_batch_dim(query)

    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    key = attn.to_k(encoder_hidden_states)

    value = attn.to_v(encoder_hidden_states)

    key = attn.head_to_batch_dim(key)

    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)

    hidden_states = ttl.tensor.bmm(attention_probs, value)

    hidden_states = attn.batch_to_head_dim(hidden_states)
    hidden_states = attn.to_out(hidden_states)
    return hidden_states
