# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn
from typing import Any, Optional, Tuple, Union
from loguru import logger

import ttnn

# logger = logging.get_logger(__name__)


class ttnn_CLIPAttention:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = ttnn.linear
        self.v_proj = ttnn.linear
        self.q_proj = ttnn.linear
        self.out_proj = ttnn.linear

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # This __call__ is not invoked
    def __call__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class ttnn_CLIPSdpaAttention(ttnn_CLIPAttention):
    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        causal_attention_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = False,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "CLIPModel is using CLIPSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
                "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
                'be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().__call__(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        bsz, tgt_len, embed_dim = hidden_states.shape

        query_states = self.q_proj(
            hidden_states,
            parameters["q_proj"]["weight"],
            bias=parameters["q_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        key_states = self.k_proj(
            hidden_states,
            parameters["k_proj"]["weight"],
            bias=parameters["k_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value_states = self.v_proj(
            hidden_states,
            parameters["v_proj"]["weight"],
            bias=parameters["v_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        query_states = ttnn.reshape(query_states, (bsz, -1, self.num_heads, self.head_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        # key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = ttnn.reshape(key_states, (bsz, -1, self.num_heads, self.head_dim))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))
        # value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = ttnn.reshape(value_states, (bsz, -1, self.num_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        query_states = ttnn.to_memory_config(query_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key_states = ttnn.to_memory_config(key_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value_states = ttnn.to_memory_config(value_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # attn_mask
        attn_mask = ttnn.to_memory_config(attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_mask, is_causal=False
        )
        # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=attn_mask,
        #     scale=self.scale,
        # )

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(
            attn_output,
            parameters["out_proj"]["weight"],
            bias=parameters["out_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return attn_output, None
