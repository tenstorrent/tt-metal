import tt_lib
import torch
import torch.nn as nn

from transformers import WhisperConfig
from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
    create_padded_tensor,
)

from python_api_testing.models.whisper.whisper_attention import TtWhisperAttention
from tt_lib.fused_ops.layernorm import Layernorm as TtLayernorm
from tt_lib.fused_ops.linear import Linear as TtLinear


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
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.embed_dim = embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim

        self.self_attn = TtWhisperAttention(
            config=config,
            base_address=f"{base_address}.self_attn",
            state_dict=self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
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
        tt_gamma = gamma.data()  # [1, 1, 1, 1024]
        tt_beta = beta.data()

        self.self_attn_layer_norm = TtLayernorm(
            tt_gamma, tt_beta, 1e-05, 1, self.embed_dim, device, 1
        )

        # DO not use DROPOUT for now
        # self.dropout = config.dropout
        # self.activation_dropout = config.activation_dropout

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
            out_features=self.encoder_ffn_dim,
            weight=fc1_weight.data(),
            bias=fc1_bias.data(),
            device=device,
        )
        self.fc2 = TtLinear(
            in_features=self.encoder_ffn_dim,
            out_features=self.embed_dim,
            weight=fc2_weight.data(),
            bias=fc2_bias.data(),
            device=device,
        )

        gamma_1 = self.state_dict[f"{base_address}.final_layer_norm.weight"]
        gamma_1 = create_padded_tensor(
            list(gamma_1.shape),
            gamma_1,
            [1, 1, 32, gamma_1.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        beta_1 = self.state_dict[f"{base_address}.final_layer_norm.bias"]
        beta_1 = create_padded_tensor(
            list(beta_1.shape),
            beta_1,
            [1, 1, 32, beta_1.shape[-1]],
            0,
            tt_lib.device.GetHost(),
        )
        tt_gamma_1 = gamma_1.data()
        tt_beta_1 = beta_1.data()

        self.final_layer_norm = TtLayernorm(
            tt_gamma_1, tt_beta_1, 1e-05, 1, self.embed_dim, device, 1
        )
        self.clamp_value = None

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tt_lib.tensor.Tensor:
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

        H_hidden_states = 1500  # hidden_states.shape()[-2]
        hidden_states = self.self_attn_layer_norm(
            hidden_states, overrideH=H_hidden_states
        )

        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # Do not use dropout for now
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = tt_lib.tensor.add(hidden_states, residual)

        residual = hidden_states

        H_hidden_states = 1500  # hidden_states.shape()[-2]
        hidden_states = self.final_layer_norm(hidden_states, overrideH=H_hidden_states)

        hidden_states = self.fc1(hidden_states)

        hidden_states = tt_lib.tensor.gelu(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states, residual)
        hidden_states_torch = tt2torch_tensor(hidden_states)

        if hidden_states_torch.dtype == torch.float16 and (
            torch.isinf(hidden_states_torch).any()
            or torch.isnan(hidden_states_torch).any()
        ):
            if self.clamp_value is not None:
                clamp_value = self.clamp_value
            else:
                clamp_value = torch.finfo(hidden_states_torch.dtype).max - 1000
                self.clamp_value = clamp_value

            hidden_states_torch = torch.clamp(
                hidden_states_torch, min=-clamp_value, max=clamp_value
            )

        hidden_states = torch2tt_tensor(hidden_states_torch, self.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
