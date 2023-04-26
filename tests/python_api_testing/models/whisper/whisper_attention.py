import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import numpy as np

import random
from typing import Optional, Tuple, Union

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor, create_padded_tensor, create_unpadded_tensor

from libs import tt_lib as ttm

from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as Ttsoftmax

class TtWhisperAttention(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        use_torch_softmax: bool = True,
    ):
        super().__init__()

        self.device = device
        self.state_dict = state_dict

        self.use_torch_softmax = use_torch_softmax

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.base_address = base_address

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        k_proj_weight = torch2tt_tensor(state_dict[f"{base_address}.k_proj.weight"], ttm.device.GetHost())

        v_proj_weight = torch2tt_tensor(state_dict[f"{base_address}.v_proj.weight"], ttm.device.GetHost())
        v_proj_bias = torch2tt_tensor(state_dict[f"{base_address}.v_proj.bias"], ttm.device.GetHost())

        q_proj_weight = torch2tt_tensor(state_dict[f"{base_address}.q_proj.weight"], ttm.device.GetHost())
        q_proj_bias = torch2tt_tensor(state_dict[f"{base_address}.q_proj.bias"], ttm.device.GetHost())

        out_proj_weight = torch2tt_tensor(state_dict[f"{base_address}.out_proj.weight"], ttm.device.GetHost())
        out_proj_bias = torch2tt_tensor(state_dict[f"{base_address}.out_proj.bias"], ttm.device.GetHost())

        self.k_proj = TtLinear(in_features=embed_dim, out_features=embed_dim, weight=k_proj_weight.data(), bias=None, device=self.device)
        self.v_proj = TtLinear(in_features=embed_dim, out_features=embed_dim, weight=v_proj_weight.data(), bias=v_proj_bias.data(), device=self.device)
        self.q_proj = TtLinear(in_features=embed_dim, out_features=embed_dim, weight=q_proj_weight.data(), bias=q_proj_bias.data(), device=self.device)
        self.out_proj = TtLinear(in_features=embed_dim, out_features=embed_dim, weight=out_proj_weight.data(), bias=out_proj_bias.data(), device=self.device)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tt_tensor: ttm.tensor.Tensor, seq_len: int, bsz: int):
        """
        _shape function copied directly from WhisperAttention.
        For now PyTorch implementation is kept, because of H mod 32 == 0 restriction in transpose and reshape methods
        H dim equals to num_head, which is set to 6 in config for Whisper-Tiny
        """
        torch_out = tt2torch_tensor(tt_tensor)
        return torch_out.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: ttm.tensor.Tensor,
        key_value_states: Optional[ttm.tensor.Tensor] = None,
        past_key_value: Optional[Tuple[ttm.tensor.Tensor]] = None,
        attention_mask: Optional[ttm.tensor.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None, # Needs to be torch.tensor for now because of the shape[6]
        output_attentions: bool = False,
    ) -> Tuple[ttm.tensor.Tensor, Optional[ttm.tensor.Tensor], Optional[Tuple[ttm.tensor.Tensor]]]:

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        is_cross_attention = key_value_states is not None

        _, bsz, tgt_len, _, = hidden_states.shape()

        tt_q_proj_output = self.q_proj(hidden_states)

        q_proj_shape = tt_q_proj_output.shape()
        torch_q_proj_mul_const = torch.full((q_proj_shape), self.scaling)
        q_proj_mul_const = torch2tt_tensor(torch_q_proj_mul_const, self.device)

        query_states_t = ttm.tensor.mul(tt_q_proj_output, q_proj_mul_const)

        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape()[1]
        ):
            """ TODO: Test these conditions """
            # reuse k,v, cross_attentions
            #Convert to torch
            key_states = tt2torch_tensor(past_key_value[0])
            value_states = tt2torch_tensor(past_key_value[1])

        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            # torch.Size([1, 6, 1504, 64])

        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            #Convert to torch
            fisrt = tt2torch_tensor(past_key_value[0])
            second = tt2torch_tensor(past_key_value[1])

            key_states = torch.cat([fisrt, key_states], dim = -2)
            value_states = torch.cat([second, value_states], dim = -2)

        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        tt_key_states = torch2tt_tensor(key_states, self.device)
        tt_value_states = torch2tt_tensor(value_states, self.device)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_0 = tt2torch_tensor(tt_key_states)
            past_key_1 = tt2torch_tensor(tt_value_states)
            past_key_0 = torch2tt_tensor(past_key_0, self.device)
            past_key_1 = torch2tt_tensor(past_key_1, self.device)
            past_key_value = (past_key_0, past_key_1)

        proj_shape = [1, bsz * self.num_heads, -1, self.head_dim]
        query_states = self._shape(query_states_t, tgt_len, bsz) # 4d

        # Conversion query_states to TTM Tensor
        tt_query_states = torch2tt_tensor(query_states, self.device)

        # Apply reshaping
        tt_query_states = ttm.tensor.reshape(tt_query_states, *proj_shape)
        tt_key_states = ttm.tensor.reshape(tt_key_states, *proj_shape)
        tt_value_states = ttm.tensor.reshape(tt_value_states, *proj_shape)

        key_states_transposed = ttm.tensor.transpose(tt_key_states)

        src_len = tt_key_states.shape()[-2]

        attn_weights = ttm.tensor.bmm(tt_query_states, key_states_transposed)

        if attn_weights.shape() != [1, bsz * self.num_heads, tgt_len, src_len]:
            raise ValueError(
                f"Attention weights should be of size {(1, bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape()}"
            )

        if attention_mask is not None:
            if attention_mask.shape() != [bsz, 1, tgt_len, src_len]:
                raise ValueError(
                    f"Attention mask should be of size {[bsz, 1, tgt_len, src_len]}, but is {attention_mask.shape()}"
                )
            # TTM implementation. Doesn't work for now
            # attn_weights = ttm.tensor.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)

            # attn_weights = ttm.tensor.bcast(attention_mask, attn_weights, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H)
            # attn_weights = ttm.tensor.reshape(attn_weights, 1, bsz * self.num_head, tgt_len, src_len)

            torch_attn_weights = tt2torch_tensor(attn_weights)
            torch_attention_mask = tt2torch_tensor(attention_mask)
            torch_attn_weights = torch_attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + torch_attention_mask
            torch_attn_weights = torch_attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights = torch2tt_tensor(torch_attn_weights, self.device)
            # print("Return from attention mask")
            # return attn_weights # PCC = 1.0

        if (not self.is_decoder) or (self.is_decoder and "encoder_attn" in self.base_address):
            # Unpad and pad with high negative value
            if self.use_torch_softmax:
                pad_value = -torch.inf
            else:
                pad_value = -100000

            input_tensors_shape = attn_weights.shape()
            input_tensors_shape[-1] = 1500

            if not "encoder_attn" in self.base_address: # encoder last hidden states [1, 1, 1500, 384]
                input_tensors_shape[-2] = 1500

            attn_weights = create_unpadded_tensor(attn_weights, input_tensors_shape)

            output_tensor_shape = attn_weights.shape()
            output_tensor_shape[-1] = 1504

            if not "encoder_attn" in self.base_address:
                output_tensor_shape[-2] = 1504

            attn_weights = create_padded_tensor(attn_weights.shape(), attn_weights, output_tensor_shape, pad_value= pad_value, device=self.device)

            if not self.use_torch_softmax:
                # pad with -10^5
                attn_weights = Ttsoftmax(attn_weights)

            else:
                # torch softmax
                attn_weights = tt2torch_tensor(attn_weights)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                attn_weights = torch2tt_tensor(attn_weights, self.device)

        else:
            #Case when padding is not used before softmax
            if self.use_torch_softmax:
                attn_weights = tt2torch_tensor(attn_weights)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                attn_weights = torch2tt_tensor(attn_weights, self.device)
            else:
                attn_weights = Ttsoftmax(attn_weights)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            # Impelement this part in torch because of view(1, -1, 1, 1) is not allowed in TT Metal reshape
            attn_weights_t = tt2torch_tensor(attn_weights)
            attn_weights_t = layer_head_mask.view(1, -1, 1, 1) * attn_weights_t.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights_t = attn_weights_t.view(bsz * self.num_heads, tgt_len, src_len)
            # Convert to TTM Tensor
            attn_weights = torch2tt_tensor(attn_weights_t, self.device)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_w_torch = tt2torch_tensor(attn_weights)
            attn_weights_reshaped_torch = attn_w_torch.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights_reshaped = torch2tt_tensor(attn_weights_reshaped_torch, self.device)
        else:
            attn_weights_reshaped = None

        """
        TODO: Dropout
        """
        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_probs = attn_weights

        attn_output = ttm.tensor.bmm(attn_probs, tt_value_states)

        if attn_output.shape() != [1, bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape()}"
            )

        attn_output = ttm.tensor.reshape(attn_output, bsz, self.num_heads, tgt_len, self.head_dim)

        # Transposing whith these dims had to be done in Pytorch
        torch_attn_output = tt2torch_tensor(attn_output)
        torch_attn_output = torch_attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        torch_attn_output = torch_attn_output.reshape(1, bsz, tgt_len, self.embed_dim)
        attn_output = torch2tt_tensor(torch_attn_output, self.device)

        attn_output = ttm.tensor.reshape(attn_output, 1, bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
