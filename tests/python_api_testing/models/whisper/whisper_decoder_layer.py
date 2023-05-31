import torch
import torch.nn as nn
import tt_lib
from typing import Optional, Tuple, Union

from transformers import WhisperConfig

from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
    create_padded_tensor,
)
from python_api_testing.models.whisper.whisper_attention import TtWhisperAttention

from tt_lib.fused_ops.layernorm import Layernorm as TtLayernorm
from tt_lib.fused_ops.linear import Linear as TtLinear


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
    ):
        super().__init__()

        self.device = device
        self.config = config
        self.state_dict = state_dict

        self.embed_dim = embed_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        # Do not use dropout for now
        self.dropout = config.dropout

        self.self_attn = TtWhisperAttention(
            config=config,
            base_address=f"{base_address}.self_attn",
            state_dict=self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            is_decoder=True,
        )

        gamma = self.state_dict[f"{base_address}.self_attn_layer_norm.weight"]
        gamma = create_padded_tensor(
            list(gamma.shape),
            gamma,
            [1, 1, 32, gamma.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        beta = self.state_dict[f"{base_address}.self_attn_layer_norm.bias"]
        beta = create_padded_tensor(
            list(beta.shape),
            beta,
            [1, 1, 32, beta.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        tt_gamma = gamma.data()
        tt_beta = beta.data()

        self.self_attn_layer_norm = TtLayernorm(
            tt_gamma, tt_beta, 1e-05, self.embed_dim, self.embed_dim, device, 1
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

        gamma1 = self.state_dict[f"{base_address}.encoder_attn_layer_norm.weight"]
        gamma1 = create_padded_tensor(
            list(gamma1.shape),
            gamma1,
            [1, 1, 32, gamma1.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        beta1 = self.state_dict[f"{base_address}.encoder_attn_layer_norm.bias"]
        beta1 = create_padded_tensor(
            list(beta1.shape),
            beta1,
            [1, 1, 32, beta1.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        tt_gamma1 = gamma1.data()
        tt_beta1 = beta1.data()
        self.encoder_attn_layer_norm = TtLayernorm(
            tt_gamma1, tt_beta1, 1e-05, 1, self.embed_dim, device, 1
        )

        fc1_weight = torch2tt_tensor(
            self.state_dict[f"{base_address}.fc1.weight"], tt_lib.device.GetHost()
        )
        fc1_bias = state_dict[f"{base_address}.fc1.bias"]
        fc1_bias = create_padded_tensor(
            list(fc1_bias.shape),
            fc1_bias,
            [1, 1, 32, fc1_bias.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        fc2_weight = torch2tt_tensor(
            self.state_dict[f"{base_address}.fc2.weight"], tt_lib.device.GetHost()
        )
        fc2_bias = state_dict[f"{base_address}.fc2.bias"]
        fc2_bias = create_padded_tensor(
            list(fc2_bias.shape),
            fc2_bias,
            [1, 1, 32, fc2_bias.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )

        self.fc1 = TtLinear(
            in_features=self.embed_dim,
            out_features=self.decoder_ffn_dim,
            weight=fc1_weight.data(),
            bias=fc1_bias.data(),
            device=device,
        )
        self.fc2 = TtLinear(
            in_features=self.decoder_ffn_dim,
            out_features=self.embed_dim,
            weight=fc2_weight.data(),
            bias=fc2_bias.data(),
            device=device,
        )

        gamma2 = self.state_dict[f"{base_address}.final_layer_norm.weight"]
        gamma2 = create_padded_tensor(
            list(gamma2.shape),
            gamma2,
            [1, 1, 32, gamma2.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        beta2 = self.state_dict[f"{base_address}.final_layer_norm.bias"]
        beta2 = create_padded_tensor(
            list(beta2.shape),
            beta2,
            [1, 1, 32, beta2.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        tt_gamma2 = gamma2.data()
        tt_beta2 = beta2.data()
        self.final_layer_norm = TtLayernorm(
            tt_gamma2, tt_beta2, 1e-05, 1, self.embed_dim, device, 1
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        encoder_hidden_states: Optional[tt_lib.tensor.Tensor] = None,
        encoder_attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # returns tt_lib.tensor.Tensor or tuple of tensors and tuples
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

        H_hidden_states = hidden_states.shape()[-2]

        hidden_states = self.self_attn_layer_norm(
            hidden_states, overrideH=H_hidden_states
        )

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

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
        hidden_states = tt_lib.tensor.add(hidden_states, residual)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states
            H_hidden_states = hidden_states.shape()[-2]
            hidden_states = self.encoder_attn_layer_norm(
                hidden_states, overrideH=H_hidden_states
            )

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
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

            hidden_states = tt_lib.tensor.add(hidden_states, residual)

            # concatenation of two tuples
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        H_hidden_states = hidden_states.shape()[-2]
        hidden_states = self.final_layer_norm(hidden_states, overrideH=H_hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = tt_lib.tensor.gelu(hidden_states)

        # TODO: When implement training
        # hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)

        # TODO: When implement training
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = tt_lib.tensor.add(residual, hidden_states)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
