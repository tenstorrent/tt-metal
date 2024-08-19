# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Optional, Tuple

import ttnn
from models.utility_functions import torch_to_tt_tensor_rm
from models.helper_funcs import Linear
from models.experimental.trocr.tt.trocr_attention import TtTrOCRAttention
from models.experimental.trocr.tt.trocr_configuration import TtTrOCRConfig


class TtTrOCRDecoderLayer(nn.Module):
    def __init__(
        self,
        config: TtTrOCRConfig,
        base_address=None,
        state_dict=None,
        device=None,
        host=None,
    ):
        super().__init__()
        self.host = host
        self.device = device
        self.embed_dim = config.hidden_size

        self.self_attn = TtTrOCRAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            base_address=f"{base_address}.self_attn",
            state_dict=state_dict,
            device=device,
            host=host,
            is_decoder=True,
        )
        self.activation_fn = ttnn.gelu

        self.self_attn_layer_norm_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.self_attn_layer_norm.weight"],
            self.device,
            put_on_device=True,
        )

        self.self_attn_layer_norm_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.self_attn_layer_norm.bias"],
            self.device,
            put_on_device=True,
        )

        self.self_attn_layer_norm = ttnn.layer_norm

        if config.is_decoder:
            self.encoder_attn = TtTrOCRAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                kdim=config.cross_attention_hidden_size,
                vdim=config.cross_attention_hidden_size,
                state_dict=state_dict,
                device=device,
                host=host,
                base_address=f"{base_address}.encoder_attn",
                is_decoder=True,
                is_cross_attention=True,
            )

            self.encoder_attn_layer_norm_weight = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.self_attn_layer_norm.weight"],
                self.device,
                put_on_device=True,
            )

            self.encoder_attn_layer_norm_bias = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.self_attn_layer_norm.bias"],
                self.device,
                put_on_device=True,
            )

            self.encoder_attn_layer_norm = ttnn.layer_norm

        self.fc1_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.fc1.weight"], self.device, put_on_device=False
        )
        self.fc1_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc1.bias"], self.device, put_on_device=False)
        self.fc1 = Linear(self.embed_dim, config.decoder_ffn_dim, self.fc1_weight, self.fc1_bias)

        self.fc2_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.fc2.weight"], self.device, put_on_device=False
        )
        self.fc2_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc2.bias"], self.device, put_on_device=False)
        self.fc2 = Linear(config.decoder_ffn_dim, self.embed_dim, self.fc2_weight, self.fc2_bias)

        self.final_layer_norm_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.final_layer_norm.weight"],
            self.device,
            put_on_device=True,
        )
        self.final_layer_norm_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.final_layer_norm.bias"],
            self.device,
            put_on_device=True,
        )
        self.final_layer_norm = ttnn.layer_norm

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        layer_head_mask: Optional[ttnn.Tensor] = None,
        cross_attn_layer_head_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> ttnn.Tensor:
        residual = hidden_states

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

        hidden_states = ttnn.add(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=self.self_attn_layer_norm_weight,
            bias=self.self_attn_layer_norm_bias,
        )

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states

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

            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(
                hidden_states,
                epsilon=1e-05,
                weight=self.encoder_attn_layer_norm_weight,
                bias=self.encoder_attn_layer_norm_bias,
            )

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)

        hidden_states = ttnn.add(residual, hidden_states)
        hidden_states = self.final_layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=self.final_layer_norm_weight,
            bias=self.final_layer_norm_bias,
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs = ttnn.add(self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
