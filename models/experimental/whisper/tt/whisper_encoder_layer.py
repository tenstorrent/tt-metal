# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import torch
import torch.nn as nn
import ttnn
from typing import Tuple

from transformers import WhisperConfig

from models.utility_functions import torch2tt_tensor, tt2torch_tensor

from models.experimental.whisper.tt.whisper_common import (
    linear,
)

# from tt_lib.fallback_ops import fallback_ops
import tt_lib.fallback_ops as fallback_ops
from models.experimental.whisper.tt.whisper_attention import TtWhisperAttention


class TtWhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        embed_dim,
        num_heads,
        encoder_ffn_dim,
        config: WhisperConfig = None,
        use_torch_gelu=True,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.embed_dim = embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.use_torch_gelu = use_torch_gelu
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.self_attn = TtWhisperAttention(
            config=config,
            base_address=f"{base_address}.self_attn",
            state_dict=self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
        )
        gamma = torch2tt_tensor(
            self.state_dict[f"{base_address}.self_attn_layer_norm.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        beta = torch2tt_tensor(
            self.state_dict[f"{base_address}.self_attn_layer_norm.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        self.self_attn_layer_norm = partial(ttnn.layer_norm, weight=gamma, bias=beta, epsilon=1e-05)

        # self.self_attn_layer_norm = fallback_ops.LayerNorm(
        #     gamma, beta, eps=1e-05, normalized_shape=self.embed_dim
        # )

        # DO not use DROPOUT for now
        # self.dropout = config.dropout
        # self.activation_dropout = config.activation_dropout

        self.fc1_weight = torch2tt_tensor(
            self.state_dict[f"{base_address}.fc1.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.fc1_bias = torch2tt_tensor(
            state_dict[f"{base_address}.fc1.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        self.fc2_weight = torch2tt_tensor(
            self.state_dict[f"{base_address}.fc2.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.fc2_bias = torch2tt_tensor(
            state_dict[f"{base_address}.fc2.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        gamma_1 = torch2tt_tensor(
            self.state_dict[f"{base_address}.final_layer_norm.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        beta_1 = torch2tt_tensor(
            self.state_dict[f"{base_address}.final_layer_norm.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        self.final_layer_norm = partial(ttnn.layer_norm, gamma=gamma_1, beta=beta_1, eps=1e-05)

        # self.final_layer_norm = fallback_ops.LayerNorm(
        #     gamma_1, beta_1, eps=1e-05, normalized_shape=self.embed_dim
        # )
        self.clamp_value = None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        layer_head_mask: ttnn.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[ttnn.Tensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # TODO: Do not use dropout for now
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = ttnn.add(hidden_states, residual)
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = linear(hidden_states, self.fc1_weight, self.fc1_bias)
        if self.use_torch_gelu:
            torch_hidden_states = tt2torch_tensor(hidden_states)
            torch_hidden_states = torch.nn.functional.gelu(torch_hidden_states)
            hidden_states = torch2tt_tensor(torch_hidden_states, self.device, ttnn.ROW_MAJOR_LAYOUT)
        else:
            hidden_states = ttnn.gelu(hidden_states)

        hidden_states = linear(hidden_states, self.fc2_weight, self.fc2_bias)
        hidden_states = ttnn.add(hidden_states, residual)

        hidden_states_torch = tt2torch_tensor(hidden_states)

        if hidden_states_torch.dtype == torch.float16 and (
            torch.isinf(hidden_states_torch).any() or torch.isnan(hidden_states_torch).any()
        ):
            if self.clamp_value is not None:
                clamp_value = self.clamp_value
            else:
                clamp_value = torch.finfo(hidden_states_torch.dtype).max - 1000
                self.clamp_value = clamp_value

            hidden_states_torch = torch.clamp(hidden_states_torch, min=-clamp_value, max=clamp_value)

        hidden_states = torch2tt_tensor(hidden_states_torch, self.device, ttnn.ROW_MAJOR_LAYOUT)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
