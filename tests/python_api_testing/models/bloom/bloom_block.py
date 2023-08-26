import torch
import math
from torch.nn import functional as F
from torch.nn import LayerNorm

from functools import partial
import tt_lib as ttl
from tt_lib.fallback_ops import fallback_ops
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_mlp as bloom_mlp
from typing import Optional, Tuple, Union
from tt_models.utility_functions import pad_by_zero

# class BloomBlock(nn.Module):
#     def __init__(self, config: BloomConfig):
#         super().__init__()
#         hidden_size = config.hidden_size

#         self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
#         self.num_heads = config.n_head
#         self.self_attention = BloomAttention(config)
#         self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

#         self.mlp = BloomMLP(config)

#         self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
#         self.hidden_dropout = config.hidden_dropout

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
#         output = self.mlp(layernorm_output, residual)

#         if use_cache:
#             outputs = (output,) + outputs
#         else:
#             outputs = (output,) + outputs[1:]

#         return outputs  # hidden_states, present, attentions


class TtBloomBlock(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.hidden_size = config.hidden_size  # 1024
        self.num_heads = config.n_head
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

        self.beta = pad_by_zero(
            state_dict[f"{base_address}.input_layernorm.bias"], device
        )[0]
        self.gamma = pad_by_zero(
            state_dict[f"{base_address}.input_layernorm.weight"], device
        )[0]
        self.input_layernorm = partial(
            ttl.tensor.layernorm,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.layer_norm_epsilon,
        )

        self.self_attention = bloom_attention.TtBloomAttention(
            config, state_dict, f"{base_address}.self_attention", device
        )

        self.beta_2 = pad_by_zero(
            state_dict[f"{base_address}.post_attention_layernorm.bias"], device
        )[0]
        self.gamma_2 = pad_by_zero(
            state_dict[f"{base_address}.post_attention_layernorm.weight"], device
        )[0]
        self.post_attention_layernorm = partial(
            ttl.tensor.layernorm,
            gamma=self.gamma_2,
            beta=self.beta_2,
            eps=self.layer_norm_epsilon,
        )

        self.mlp = bloom_mlp.TtBloomMLP(
            config, state_dict, f"{base_address}.mlp", device
        )

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
        layernorm_output = self.input_layernorm(
            hidden_states
        )  # , overrideH=hidden_states.shape()[-2])

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
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

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(
            attention_output
        )  # , overrideH=attention_output.shape()[-2])

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP
        output = self.mlp(layernorm_output, residual, device)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
