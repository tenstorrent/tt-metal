import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_mlp as bloom_mlp

from fused_ops.linear import Linear as TtLinear
from fused_ops.layernorm import Layernorm as TtLayernorm

from fused_ops.softmax import softmax as TtSoftmax
from transformers import BloomForCausalLM

from typing import Optional, Tuple, Union


# class BloomBlock(torch.nn.Module):
#     def __init__(self, dict_name, num, hugging_bloom_reference_model, hidden_size, num_heads, layer_norm_epsilon, apply_residual_connection_post_layernorm=False, hidden_dropout=0.0, beta=0.0):
#         super().__init__()
#         self.hidden_size = hidden_size
#         state_dict = hugging_bloom_reference_model.state_dict()
#         pt_beta = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.input_layernorm.bias", state_dict)
#         pt_gamma = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.input_layernorm.weight", state_dict)

#         #pt_beta = bloom_utils.pt_load_layer_weights("transformer.h.0.input_layernorm.bias", state_dict)
#         ##pt_gamma = bloom_utils.pt_load_layer_weights("transformer.h.0.input_layernorm.weight", state_dict)
#         self.input_layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
#         #self.input_layernorm.bias = pt_beta
#         #self.input_layernorm.weight = pt_gamma


#         self.num_heads = num_heads

#         self.self_attention = bloom_attention.BloomAttention(dict_name, num, hugging_bloom_reference_model, hidden_size, num_heads, hidden_dropout, beta)


#         pt_beta_2 = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.post_attention_layernorm.bias", state_dict)
#         pt_gamma_2 = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.post_attention_layernorm.weight", state_dict)


#         self.post_attention_layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
#         #self.post_attention_layernorm.bias = pt_beta_2
#         #self.post_attention_layernorm.weight = pt_gamma_2

#         self.mlp = bloom_mlp.BloomMLP(dict_name, num, state_dict, hidden_dropout, hidden_size, False)

#         self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
#         self.hidden_dropout = hidden_dropout

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         alibi: torch.Tensor,
#         attention_mask: torch.Tensor,
#         layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         use_cache: bool = False,
#         output_attentions: bool = False,
#     ):
#         # hidden_states: [batch_size, seq_length, hidden_size]

#         # Layer norm at the beginning of the transformer layer.
#         layernorm_output = self.input_layernorm(hidden_states)

#         # Layer norm post the self attention.
#         if self.apply_residual_connection_post_layernorm:
#             residual = layernorm_output
#         else:
#             residual = hidden_states

#         # Self attention.
#         attn_outputs = self.self_attention(
#             layernorm_output,
#             residual,
#             layer_past=layer_past,
#             attention_mask=attention_mask,
#             alibi=alibi,
#             head_mask=head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#         )

#         attention_output = attn_outputs[0]

#         outputs = attn_outputs[1:]

#         layernorm_output = self.post_attention_layernorm(attention_output)

#         # Get residual
#         if self.apply_residual_connection_post_layernorm:
#             residual = layernorm_output
#         else:
#             residual = attention_output

#         # MLP.
#         outputs = self.mlp(layernorm_output, residual)

#         return outputs  # hidden_states, present, attentions


class TtBloomBlock(torch.nn.Module):
    def __init__(self, device, dict_name, num, hugging_bloom_reference_model, hidden_size, num_heads, layer_norm_epsilon, apply_residual_connection_post_layernorm=False, hidden_dropout=0.0, beta=0.0):

        super().__init__()
        self.hidden_size = hidden_size

        state_dict = hugging_bloom_reference_model.state_dict()

        tt_beta = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.input_layernorm.bias", state_dict)
        tt_gamma = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.input_layernorm.weight", state_dict)

        self.input_layernorm = TtLayernorm(tt_gamma, tt_beta, layer_norm_epsilon, self.hidden_size, self.hidden_size, device, 1)
        self.hidden_dropout = hidden_dropout
        self.num_heads = num_heads

        self.self_attention = bloom_attention.TtBloomAttention(device, dict_name, num, hugging_bloom_reference_model, self.hidden_size, self.num_heads, hidden_dropout, beta)
        tt_beta_2 = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.post_attention_layernorm.bias", state_dict)
        tt_gamma_2 = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.post_attention_layernorm.weight", state_dict)

        self.post_attention_layernorm = TtLayernorm(tt_gamma_2, tt_beta_2, layer_norm_epsilon, self.hidden_size, self.hidden_size, device, 1)
        self.mlp = bloom_mlp.TtBloomMLP(device, dict_name, num, hugging_bloom_reference_model, self.hidden_dropout, self.hidden_size, False)
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm

    def forward(
        self,
        device,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        tt_hidden_states = bloom_utils.torch2tt_tensor(hidden_states, device)
        tt_layernorm_output = self.input_layernorm(tt_hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            tt_residual = tt_layernorm_output
        else:
            tt_residual = tt_hidden_states

        layernorm_output = bloom_utils.tt2torch_tensor(tt_layernorm_output)
        residual = bloom_utils.tt2torch_tensor(tt_residual)

        # Self attention.
        tt_attn_outputs = self.self_attention(
            device,
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attn_outputs = bloom_utils.tt2torch_tensor(tt_attn_outputs)

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        tt_attention_output = bloom_utils.torch2tt_tensor(attention_output, device)
        tt_layernorm_output = self.post_attention_layernorm(tt_attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            tt_residual = tt_layernorm_output
        else:
            tt_residual = tt_attention_output

        residual = bloom_utils.tt2torch_tensor(tt_residual)
        layernorm_output = bloom_utils.tt2torch_tensor(tt_layernorm_output)

        # MLP.
        tt_output = self.mlp(layernorm_output, residual, device)

        #pt_outputs = bloom_utils.tt2torch_tensor(tt_output)
        #pt_outputs  = pt_outputs[1:]
        #tt_outputs = bloom_utils.torch2tt_tensor(pt_outputs, device)

        """
        if use_cache:
            final_outputs = ttm.tensor.add(tt_output, tt_outputs)
        else:
            outputs_1 = outputs[1:]
            tt_outputs_1 = torch2tt_tensor(outputs_1, device)

            final_outputs = ttm.tensor.add(tt_output, tt_outputs_1)

        return final_outputs  # hidden_states, present, attentions
        """

        return tt_output
