from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn
import torch
from libs.tt_lib.utils import tilize_to_list, pad_weight

from libs import tt_lib as ttl
from utility_functions import print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.fused_ops.linear import Linear as tt_linear
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc

from typing import Optional, Tuple, Union
from transformers import CLIPModel


class TtCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, device, state_dict, config=None, hidden_size=None, num_attention_heads=None, host=None, base_address="text_model.encoder.layers.10.self_attn"):
        super().__init__()
        self.config = config
        self.device = device
        self.embed_dim = config.hidden_size if config else hidden_size
        self.num_heads = config.num_attention_heads if config else num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        # self.dropout = config.attention_dropout

        self.k_proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.k_proj.weight"]))
        self.k_proj_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.k_proj.bias"]))
        self.k_proj = tt_linear(self.embed_dim, self.embed_dim, self.k_proj_weights, self.k_proj_bias, device)


        self.v_proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.v_proj.weight"]))
        self.v_proj_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.v_proj.bias"]))
        self.v_proj = tt_linear(self.embed_dim, self.embed_dim, self.v_proj_weights, self.v_proj_bias, device)

        self.q_proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.q_proj.weight"]))
        self.q_proj_bias = tilize_to_list(pad_weight(state_dict[ f"{base_address}.q_proj.bias"]))
        self.q_proj = tt_linear(self.embed_dim, self.embed_dim, self.q_proj_weights, self.q_proj_bias, device)

        self.out_proj_weights = tilize_to_list(pad_weight(state_dict[ f"{base_address}.out_proj.weight"]))
        self.out_proj_bias = tilize_to_list(pad_weight(state_dict[ f"{base_address}.out_proj.bias"]))
        self.out_proj = tt_linear(self.embed_dim, self.embed_dim, self.out_proj_weights, self.out_proj_bias, device)


    def _shape(self, tensor, seq_len: int, bsz: int):
        t = ttl.tensor.reshape(tensor, bsz, seq_len, self.num_heads, self.head_dim) .transpose(1, 2).contiguous()
        tt = ttl.tensor.transpose_hc(t)
        return t


    def forward(
        self,
        hidden_states,
        attention_mask,
        causal_attention_mask = None,
        output_attentions: Optional[bool] = False,):
        """Input shape: 1 x Batch x Time x Channel"""

        N, bsz, tgt_len, embed_dim = hidden_states.shape()

        scale = torch.full(hidden_states.shape(), self.scale)
        scale = torch_to_tt_tensor(scale, self.device)
        query_states = self.q_proj(hidden_states)
        query_states = ttl.tensor.mul(scale, query_states)

        t_k_proj = self.k_proj(hidden_states)
        key_states = self._shape(t_k_proj, -1, bsz)

        t_v_proj = self.v_proj(hidden_states)
        value_states = self._shape(t_v_proj, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = ttl.tensor.reshape(query_states, *proj_shape)

        key_states = ttl.tensor.reshape(key_states, *proj_shape)
        value_states = ttl.tensor.reshape(value_states, *proj_shape)

        src_len = key_states.shape()[1]

        T_key_states = ttl.tensor.transpose(key_states)
        attn_weights = ttl.tensor.bmm(query_states, key_states)

        if attn_weights.shape() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if causal_attention_mask is not None:
            if causal_attention_mask.shape() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = ttl.tensor.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)
            attn_weights = ttl.tensor.add(attn_weights, causal_attention_mask)

            attn_weights = ttl.tensor.reshape(attn_weights, 1, bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )

            attn_weights = ttl.tensor.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)
            attn_weights = ttl.tensor.add(attn_weights, attention_mask)

            attn_weights = ttl.tensor.reshape(attn_weights, 1, bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ttl.fused_ops.softmax(attn_weights)

        # intentionally ignoring output_attention line 103 to 111 and return arg
        # since it is not used

        # intentionally ignoring dropout since it does nothing in inference
        attn_probls = attn_weights # dropout

        attn_output = ttl.tensor.bmm(attn_probs, value_states)

        if attn_output.shape() != (1, bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = ttl.tensor.reshape(attn_output, bsz, self.num_heads, tgt_len, self.head_dim)

        attn_output = ttl.tensor.transpose_hc(attn_output)
        attn_output = ttl.tensor.reshape(1, bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped # output_attention is ignored since it is false and makes no difference in our case


def run_clip_attention_inference(device, host):
    # hidden_states = 1, 77, 768
    #     attention_mask = None,
    #     causal_attention_mask  = (1, 1, 77, 77)
    #     output_attentions = False
    # change all 77 to 96

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
    model.eval()
    state_dict = model.state_dict()
    config = model.config.text_config

    num_attention_heads = 1
    hidden_size = config.hidden_size
    D = 96 # real value is 77
    embed_dim = config.hidden_size

    hidden_state_shape = [1, 1, D, config.hidden_size]
    causal_attention_mask_shape = (1, 1, D, D)



    hidden_states = torch.randn(hidden_state_shape)
    attention_mask = None
    causal_attention_mask = torch.randn(causal_attention_mask_shape)
    output_attentions = False


    torch_ca = model.text_model.encoder.layers[10].self_attn
    print(torch_ca)
    torch_out = torch_ca(hidden_states=hidden_states.squeeze(0), causal_attention_mask=causal_attention_mask, attention_mask=None)


    tt_hidden_states = torch_to_tt_tensor(hidden_states, device)
    tt_causal_attention_mask = torch_to_tt_tensor(causal_attention_mask, device)


    tt_ca = TtCLIPAttention(device=device, config=config, state_dict=state_dict)

    tt_out = tt_ca(hidden_states=tt_hidden_states, causal_attention_mask=tt_causal_attention_mask, attention_mask=None)

    tt_output = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_output, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    run_clip_attention_inference(device, host)
    ttl.device.CloseDevice(device)
