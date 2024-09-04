# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import torch
import torch.nn as nn

import ttnn
from typing import Optional, Tuple, Union

from transformers import WhisperConfig

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.experimental.whisper.tt.whisper_common import linear
from models.experimental.whisper.tt.whisper_attention import TtWhisperAttention


class TtWhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        embed_dim,
        num_heads,
        decoder_ffn_dim,
        config: WhisperConfig = None,
        use_torch_gelu=True,
    ):
        super().__init__()

        self.device = device
        self.config = config
        self.state_dict = state_dict
        self.use_torch_gelu = use_torch_gelu

        self.embed_dim = embed_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        # Do not use dropout for now
        # self.dropout = config.dropout

        self.self_attn = TtWhisperAttention(
            config=config,
            base_address=f"{base_address}.self_attn",
            state_dict=self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            is_decoder=True,
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

        self.self_attn_layer_norm = partial(
            ttnn.layer_norm,
            weight=gamma,
            bias=beta,
            epsilon=1e-05,
        )

        self.encoder_attn = TtWhisperAttention(
            config=config,
            base_address=f"{base_address}.encoder_attn",
            state_dict=self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            is_decoder=True,
        )

        gamma1 = torch2tt_tensor(
            self.state_dict[f"{base_address}.encoder_attn_layer_norm.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        beta1 = torch2tt_tensor(
            self.state_dict[f"{base_address}.encoder_attn_layer_norm.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        self.encoder_attn_layer_norm = partial(ttnn.layer_norm, weight=gamma1, bias=beta1, epsilon=1e-05)

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

        gamma2 = torch2tt_tensor(
            self.state_dict[f"{base_address}.final_layer_norm.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        beta2 = torch2tt_tensor(
            self.state_dict[f"{base_address}.final_layer_norm.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.final_layer_norm = partial(ttnn.layer_norm, weight=gamma2, bias=beta2, epsilon=1e-05)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[ttnn.Tensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # TODO: When implement training
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = ttnn.add(hidden_states, residual)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            # TODO: When implement training
            # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            hidden_states = ttnn.add(hidden_states, residual)

            # concatenation of two tuples
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = linear(hidden_states, self.fc1_weight, self.fc1_bias)

        # Use torch gelu bc ttlib gelu drops pcc:
        if self.use_torch_gelu:
            torch_hidden_states = tt2torch_tensor(hidden_states)
            torch_hidden_states = torch.nn.functional.gelu(torch_hidden_states)
            hidden_states = torch2tt_tensor(torch_hidden_states, self.device, ttnn.ROW_MAJOR_LAYOUT)
        else:
            hidden_states = ttnn.gelu(hidden_states)

        # TODO: When implement training
        # hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = linear(hidden_states, self.fc2_weight, self.fc2_bias)

        # TODO: When implement training
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = ttnn.add(residual, hidden_states)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
