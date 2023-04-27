from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.baddbmm as baddbmm
import python_api_testing.models.bloom.dropout_add as dropout_add

import python_api_testing.models.bloom.bloom_attention_merge_heads as bloom_attention_merge_heads

from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TtSoftmax
from transformers import BloomForCausalLM

from typing import Optional, Tuple, Union


def split_heads(fused_qkv: torch.Tensor, num_heads, head_dim) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `fused_qkv`

    Args:
        fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    if(len(fused_qkv.shape)==3):
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    else:
        _, batch_size, seq_length, three_times_hidden_size = fused_qkv.shape

    fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads, 3, head_dim)
    return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

def merge_heads(x: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
    """
    Merge heads together over the last dimenstion

    Args:
        x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    # First view     to decompose the batch size
    # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
    x = x.view(batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, num_heads * head_dim)

class BloomAttention(torch.nn.Module):
    def __init__(self, dict_name, num, bloom_reference_model, hidden_size, num_heads, hidden_dropout, beta):
        super().__init__()

        sd = bloom_reference_model.state_dict()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = beta

        weight_q = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.query_key_value.weight", sd)
        bias_q = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.query_key_value.bias", sd)

        weight_d = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.dense.weight", sd)
        bias_d = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.dense.bias", sd)

        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = torch.nn.Dropout(self.hidden_dropout)

        self.query_key_value.weight = weight_q
        self.query_key_value.bias = bias_q

        self.dense.weight = weight_d
        self.dense.bias = bias_d

        self.pretraining_tp = False
        self.slow_but_exact = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)


        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11

        print('PT ALIBI SHAPE ------')
        print(alibi.shape)


        print('ALIBI:')
        print(alibi.shape)

        print('BATCH1:')
        print(query_layer.shape)

        print('BATCH2:')
        print(key_layer.shape)

        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = merge_heads(context_layer, self.num_heads, self.head_dim)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add.dropout_add(output_tensor, residual, self.hidden_dropout, self.training)


        return output_tensor



class TtBloomAttention(torch.nn.Module):
    def __init__(self, device, dict_name, num, bloom_reference_model, hidden_size, num_heads, hidden_dropout, beta):
        super().__init__()

        sd = bloom_reference_model.state_dict()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = hidden_size
        self.hidden_dropout = hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        weight_q = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.self_attention.query_key_value.weight", sd)
        bias_q = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.self_attention.query_key_value.bias", sd)

        weight_d = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.self_attention.dense.weight", sd)
        bias_d = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.self_attention.dense.bias", sd)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = beta

        self.query_key_value = TtLinear(self.hidden_size, 3 * self.hidden_size, weight_q, bias_q, device)
        self.dense = TtLinear(self.hidden_size, self.hidden_size, weight_d, bias_d, device)
        self.attention_dropout = torch.nn.Dropout(0.0)


    def forward(
        self,
        device,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):


        tt_hidden_states = bloom_utils.torch2tt_tensor(hidden_states, device)

        tt_fused_qkv = self.query_key_value(tt_hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        fused_qkv = bloom_utils.tt2torch_tensor(tt_fused_qkv)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)

        batch_size, q_length, _, _ = query_layer.shape

        #p_reshaped_query_layer = torch.Tensor(fused_qkv).reshape(1, batch_size, seq * self.num_heads,  q_length, self.head_dim)
        #query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        tt_query_layer = bloom_utils.torch2tt_tensor(query_layer, device)

        tt_transposed_query_layer = ttm.tensor.transpose(tt_query_layer)
        tt_reshaped_query_layer = ttm.tensor.reshape(tt_transposed_query_layer, 1, batch_size * self.num_heads, q_length, self.head_dim)

        #key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        key_layer = key_layer.permute(0, 2, 3, 1)

        tt_key_layer = bloom_utils.torch2tt_tensor(key_layer, device)
        tt_reshaped_key_layer = ttm.tensor.reshape(tt_key_layer, 1, batch_size * self.num_heads, self.head_dim, q_length)

        #value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        tt_value_layer = bloom_utils.torch2tt_tensor(value_layer, device)
        tt_transposed_value_layer = ttm.tensor.transpose(tt_value_layer)
        tt_reshaped_value_layer = ttm.tensor.reshape(tt_transposed_value_layer, 1, batch_size * self.num_heads, q_length, self.head_dim)

        p_reshaped_key_layer = bloom_utils.tt2torch_tensor(tt_reshaped_key_layer)
        p_reshaped_query_layer = bloom_utils.tt2torch_tensor(tt_reshaped_query_layer)

        p_reshaped_query_layer_squeezed = p_reshaped_query_layer.squeeze()
        p_reshaped_key_layer_squeezed = p_reshaped_key_layer.squeeze()

        _, _, kv_length = p_reshaped_key_layer_squeezed.shape


        print('ALIBI:')
        print(alibi.shape)

        print('BATCH1:')
        print(p_reshaped_query_layer_squeezed.shape)

        print('BATCH2:')
        print(p_reshaped_key_layer_squeezed.shape)



        tt_matmul_result = baddbmm.tt_baddbmm(device=device, input=alibi, batch1=p_reshaped_query_layer_squeezed, batch2=p_reshaped_key_layer_squeezed, beta=self.beta, alpha=self.inv_norm_factor)

        # change view to [batch_size, num_heads, q_length, kv_length]
        tt_attention_scores = ttm.tensor.reshape(tt_matmul_result, batch_size, self.num_heads, q_length, kv_length)
        attention_scores = bloom_utils.tt2torch_tensor(tt_attention_scores)

        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)

        tt_attn_weights = bloom_utils.torch2tt_tensor(attn_weights, device)

        tt_attention_probs = TtSoftmax(tt_attn_weights)


        if head_mask is not None:
            tt_head_mask =  bloom_utils.torch2tt_tensor(head_mask, device)
            tt_attention_probs = ttm.mul(tt_attention_probs, head_mask)

        # change view [batch_size x num_heads, q_length, kv_length]
        tt_attention_probs_reshaped = ttm.tensor.reshape(tt_attention_probs, 1, batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]

        tt_context_layer = ttm.tensor.bmm(tt_attention_probs_reshaped, tt_reshaped_value_layer)

        pt_context_layer = bloom_utils.tt2torch_tensor(tt_context_layer)

        #tt_context_layer = ttm.tensor.matmul(tt_attention_probs_reshaped, tt_reshaped_value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]

        merged_context_layer = bloom_attention_merge_heads.tt_merge_heads(pt_context_layer.squeeze(), self.num_heads, self.hidden_size, self.num_heads, device)

        output_tensor = self.dense(merged_context_layer)

        pt_output_tensor = bloom_utils.tt2torch_tensor(output_tensor)

        output_tensor = dropout_add.tt_dropout_add(pt_output_tensor, residual, self.hidden_dropout, False, device)

        #outputs = ttm.tensor.add(output_tensor, tt_attention_probs_reshaped)

        #outputs = (output_tensor, present)
        #if output_attentions:
        #    outputs += (attention_probs,)

        return output_tensor


def run_bloom_attention_inference(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)

    tt_bloom_attention = TtBloomAttention(device, "transformer.h", 0, hugging_bloom_reference_model, 1024, 32, 0.0, 0.0)
    pt_bloom_attention = BloomAttention("transformer.h",0, hugging_bloom_reference_model, 1024, 32, 0.0, 0.0)

    # Prepare input
    torch.manual_seed(0)

    hidden_states = torch.rand(1, 64, 1024)
    residual = torch.rand(1, 64, 1024)
    alibi = torch.rand(32, 64, 64)
    attention_mask = torch.randint(0, 2, (1, 1, 64, 64))

    pt_out = pt_bloom_attention.forward(hidden_states, residual, alibi, attention_mask)
    print("Finished calc pt")
    print(pt_out)

    tt_out = tt_bloom_attention.forward(device, hidden_states, residual, alibi, attention_mask)
    print("Finished calc tt")

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print(comp_allclose(pt_out.unsqueeze(0), tt_out_converted))
    print(comp_pcc(pt_out, tt_out_converted))

    #tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))


    #pt_bloom = BloomAttention(64, 8, 0.0, 0.0)
    #pytorch_out = pt_bloom.forward(hidden_states, residual, alibi, attention_mask)


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_bloom_attention_inference(device)
    ttm.device.CloseDevice(device)
