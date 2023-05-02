from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline

from typing import Optional
from libs import tt_lib as ttl
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor


from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.softmax import softmax as TtSoftmax



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
        base_address="mid_block.attentions.0.transformer_blocks.0.attn1",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.device = device
        self.host = host

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.torch_group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
            self.torch_group_norm.weight = nn.Parameter(state_dict[f"{base_address}.torch_group_norm.weight"])
            self.torch_group_norm.bias = nn.Parameter(state_dict[f"{base_address}.torch_group_norm.bias"])
        else:
            self.torch_group_norm = None

        # if cross_attention_norm:
        #     self.norm_cross = nn.LayerNorm(cross_attention_dim)

        qweights = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_q.weight"]))
        qbias = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_q.bias"])) if bias else None
        self.to_q = TtLinear(query_dim, inner_dim, qweights, qbias, self.device)
        # self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)

        kweights = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_k.weight"]))
        print("kweight", state_dict[f"{base_address}.to_k.weight"].shape)
        kbias = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_k.bias"])) if bias else None

        self.to_k = TtLinear(cross_attention_dim, inner_dim, kweights, kbias, self.device)
        # self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        vweights = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_v.weight"]))
        vbias = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_v.bias"])) if bias else None
        self.to_v = TtLinear(cross_attention_dim, inner_dim, vweights, vbias, self.device)
        # self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            add_k_proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.add_k_proj.weight"]))
            add_k_proj_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.add_k_proj.bias"])) if bias else None
            self.add_k_proj_weights = TtLinear(added_kv_proj_dim, cross_attention_dim, add_k_proj_weights, add_k_proj_bias, self.device)
            # self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

            add_v_proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.add_v_proj.weight"]))
            add_v_proj_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.add_v_proj.bias"]))
            self.add_v_proj = TtLinear(cross_attention_dim, inner_dim, add_v_proj_weights, add_v_proj_bias, self.device)
            # self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        # self.to_out = nn.ModuleList([])
        print("linear dims", inner_dim, query_dim)

        to_out0_weight = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_out.0.weight"]))
        to_out0_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.to_out.0.bias"]))
        self.to_out = TtLinear(inner_dim, query_dim, to_out0_weight, to_out0_bias, self.device)
        # self.to_out.append(self.to_out0)
        # self.to_out.append(nn.Linear(inner_dim, query_dim))

        # self.to_out.append(nn.Dropout(dropout))

        # processor = CrossAttnProcessor()

    def set_attention_slice(self, slice_size):
        assert False, "Not Implemented"
        # if slice_size is not None and slice_size > self.sliceable_head_dim:
        #     raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        # if slice_size is not None and self.added_kv_proj_dim is not None:
        #     processor = SlicedAttnAddedKVProcessor(slice_size)
        # elif slice_size is not None:
        #     processor = SlicedAttnProcessor(slice_size)
        # elif self.added_kv_proj_dim is not None:
        #     processor = CrossAttnAddedKVProcessor()
        # else:
        #     processor = CrossAttnProcessor()

        # self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `CrossAttention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return CrossAttnProcessor(self, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,)
        # return self.processor(
        #     self,
        #     hidden_states,
        #     encoder_hidden_states=encoder_hidden_states,
        #     attention_mask=attention_mask,
        #     **cross_attention_kwargs,
        # )

    def batch_to_head_dim(self, tensor):
        # used
        head_size = self.heads
        _, batch_size, seq_len, dim = tensor.shape()
        tensor = ttl.tensor.reshape(tensor, batch_size // head_size, head_size, seq_len, dim)

        tensor = tt_to_torch_tensor(tensor, self.host)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        tensor = torch_to_tt_tensor(tensor, self.device)

        return tensor

    def head_to_batch_dim(self, tensor):
        # used
        # TODO: ops; permute and reshape
        head_size = self.heads
        _, batch_size, seq_len, dim = tensor.shape()
        # tensor = ttl.tensor.reshape(tensor, batch_size, seq_len, head_size, dim // head_size)
        tensor = tt_to_torch_tensor(tensor, self.host)
        tensor = torch.reshape(tensor, (batch_size, seq_len, head_size, dim // head_size))
        # tensor = tt_to_torch_tensor(tensor, self.host)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        tensor = torch_to_tt_tensor(tensor, self.device)
        print(tensor.shape(), "line 177 in head to batch dim")
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):

        # dtype = query.dtype
        # if self.upcast_attention:
        #     query = query.float()
        #     key = key.float()
        # TODO: Aaron
        t_key = ttl.tensor.transpose(key)
        temp = ttl.tensor.bmm(query, t_key)
        scale_tensor = torch.full(temp.shape(), self.scale)
        scale_tensor = torch_to_tt_tensor(scale_tensor, self.device)
        attention_scores = ttl.tensor.mul(scale_tensor, temp)
        # attention_scores = torch.baddbmm(
        #     torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
        #     query,
        #     key.transpose(-1, -2),
        #     beta=0,
        #     alpha=self.scale,
        # )

        if attention_mask is not None:
            attention_scores = ttl.tensor.add(attention_scores, attention_mask)
            # attention_scores = attention_scores + attention_mask

        # if self.upcast_softmax:
            # attention_scores = attention_scores.float()

        # attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = TtSoftmax(attention_scores)
        # attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length):
        # used
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            # if attention_mask.device.type == "mps":
            #     # HACK: MPS: Does not support padding by greater than dimension of input tensor.
            #     # Instead, we can manually construct the padding tensor.
            #     padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
            #     padding = torch.zeros(padding_shape, device=attention_mask.device)
            #     attention_mask = torch.concat([attention_mask, padding], dim=2)
            # else:
            attention_mask = tt_to_torch_tensor(attention_mask, self.host)
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0) # TODO: MISSING OP
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
            attention_mask = torch_to_tt_tensor(attention_mask, self.device)

        return attention_mask


def CrossAttnProcessor(attn: TtCrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
    _, batch_size, sequence_length, _ = hidden_states.shape()
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

    query = attn.to_q(hidden_states)
    query = attn.head_to_batch_dim(query)

    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    # encoder_hidden_states: [1, 2, 64, 1280]
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    hidden_states = ttl.tensor.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    # hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out(hidden_states)
    # dropout
    # hidden_states = attn.to_out[1](hidden_states)

    return hidden_states
