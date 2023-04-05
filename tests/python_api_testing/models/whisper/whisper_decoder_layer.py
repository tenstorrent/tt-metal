import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import numpy as np

import random
from typing import Optional, Tuple, Union
from loguru import logger

from transformers import WhisperModel, WhisperConfig

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor

from python_api_testing.models.whisper.whisper_attention import TtWhisperAttention
from python_api_testing.fused_ops.layernorm import Layernorm as TtLayernorm

from libs import tt_lib as ttm

from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from fused_ops.linear import Linear as TtLinear

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

class TtWhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        embed_dim,
        num_heads,
        decoder_ffn_dim,
        config: WhisperConfig=None
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
            base_address=f"{base_address}.self_attn",
            state_dict = self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            is_decoder=True,
        )

        gamma = torch2tt_tensor(self.state_dict[f"{base_address}.self_attn_layer_norm.weight"], ttm.device.GetHost())
        beta = torch2tt_tensor(self.state_dict[f"{base_address}.self_attn_layer_norm.bias"], ttm.device.GetHost())
        tt_gamma = gamma.data()
        tt_beta = beta.data()

        self.self_attn_layer_norm = TtLayernorm(tt_gamma, tt_beta, 1e-05, self.embed_dim, self.embed_dim, device, 1)

        self.encoder_attn = TtWhisperAttention(
            base_address=f"{base_address}.encoder_attn",
            state_dict = self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            is_decoder=True,
        )

        gamma1 = torch2tt_tensor(self.state_dict[f"{base_address}.encoder_attn_layer_norm.weight"], ttm.device.GetHost())
        beta1 = torch2tt_tensor(self.state_dict[f"{base_address}.encoder_attn_layer_norm.bias"], ttm.device.GetHost())
        tt_gamma1 = gamma1.data()
        tt_beta1 = beta1.data()
        self.encoder_attn_layer_norm = TtLayernorm(tt_gamma1, tt_beta1, 1e-05, self.embed_dim, self.embed_dim, device, 1)

        fc1_weight = torch2tt_tensor(self.state_dict[f"{base_address}.fc1.weight"], ttm.device.GetHost())
        fc1_bias = torch2tt_tensor(self.state_dict[f"{base_address}.fc1.bias"], ttm.device.GetHost())
        fc2_weight = torch2tt_tensor(self.state_dict[f"{base_address}.fc2.weight"], ttm.device.GetHost())
        fc2_bias = torch2tt_tensor(self.state_dict[f"{base_address}.fc2.bias"], ttm.device.GetHost())

        self.fc1 = TtLinear(in_features=self.embed_dim, out_features=self.decoder_ffn_dim, weight=fc1_weight.data(), bias=fc1_bias.data(), device=device)
        self.fc2 = TtLinear(in_features=self.decoder_ffn_dim, out_features=self.embed_dim, weight=fc2_weight.data(), bias=fc2_bias.data(), device=device)

        gamma2 = torch2tt_tensor(self.state_dict[f"{base_address}.final_layer_norm.weight"], ttm.device.GetHost())
        beta2 = torch2tt_tensor(self.state_dict[f"{base_address}.final_layer_norm.bias"], ttm.device.GetHost())
        tt_gamma2 = gamma2.data()
        tt_beta2 = beta2.data()
        self.final_layer_norm = TtLayernorm(tt_gamma2, tt_beta2, 1e-05, self.embed_dim, self.embed_dim, device, 1)


    def forward(
        self,
        hidden_states: ttm.tensor.Tensor,
        attention_mask: Optional[ttm.tensor.Tensor] = None,
        encoder_hidden_states: Optional[ttm.tensor.Tensor] = None,
        encoder_attention_mask: Optional[ttm.tensor.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[ttm.tensor.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
    # returns ttm.tensor.Tensor or tuple of tensors and tuples
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
        hidden_states = self.self_attn_layer_norm(hidden_states,overrideH=H_hidden_states)

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

        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = ttm.tensor.add(hidden_states, residual)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states
            H_hidden_states = hidden_states.shape()[-2]
            hidden_states = self.encoder_attn_layer_norm(hidden_states,overrideH=H_hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            hidden_states = ttm.tensor.add(hidden_states, residual)

            # concatenation of two tuples
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        H_hidden_states = hidden_states.shape()[-2]
        hidden_states = self.final_layer_norm(hidden_states, overrideH=H_hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = ttm.tensor.gelu(hidden_states)

        # No dropout
        # hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)

        # No dropout
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = ttm.tensor.add(residual, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def run_whisper_decoder_layer():
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    state_dict = model.state_dict()

    # Accessing the model configuration
    configuration = model.config

    embed_dim = configuration.d_model # 384
    decoder_ffn_dim = configuration.decoder_ffn_dim # 1536
    num_heads = configuration.decoder_attention_heads # 6

    DECODER_IND = 0

    pytorch_model = model.decoder.layers[DECODER_IND]
    pytorch_model.eval()

    base_address = f"decoder.layers.{DECODER_IND}"

    batch = 1
    tgt_len = 32
    seq_len = 32

    hidden_state_input_tensor = torch.randn(batch, seq_len, embed_dim)
    encoder_hidden_states = torch.rand(batch, seq_len, embed_dim)

    attention_mask_input_tensor = torch.randn(batch, 1, tgt_len, seq_len)
    encoder_attention_mask = torch.rand(batch, 1, tgt_len, seq_len)
    layer_head_mask_input_tensor = torch.rand(num_heads, )
    cross_attn_layer_head_mask = torch.rand(num_heads)
    past_key_value = None

    test_with_all_inputs = True
    debug = True
    if debug:
        print("Sizes of input tensors")
        print(hidden_state_input_tensor.size())
        print(attention_mask_input_tensor.size())
        print(encoder_hidden_states.size())
        print(encoder_attention_mask.size())
        print(layer_head_mask_input_tensor.size())

    with torch.no_grad():
        if test_with_all_inputs:
            pytorch_output = pytorch_model(
                hidden_states = hidden_state_input_tensor,
                attention_mask = attention_mask_input_tensor,
                encoder_hidden_states = encoder_hidden_states,
                encoder_attention_mask = encoder_attention_mask,
                layer_head_mask = layer_head_mask_input_tensor,
                cross_attn_layer_head_mask = cross_attn_layer_head_mask,
                past_key_value = None,
                output_attentions = False,
                use_cache=configuration.use_cache,
            )
        else:
            pytorch_output = pytorch_model(
                hidden_states = hidden_state_input_tensor,
                encoder_hidden_states = encoder_hidden_states,
                output_attentions = False,
                use_cache=configuration.use_cache,
            )
        print(pytorch_output)

    """ TTM Whisper Decoder Layer """

    # Make inputs ready

    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)
    print("ttm_tensor_hidden_state size")
    print(ttm_tensor_hidden_state.shape())

    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)
    print(ttm_tensor_attention_mask.shape())

    ttm_encoder_hidden_states = torch2tt_tensor(encoder_hidden_states, device)
    print("ttm_encoder_hidden_states size")
    print(ttm_encoder_hidden_states.shape())

    ttm_encoder_attention_mask = torch2tt_tensor(encoder_attention_mask, device)
    print("ttm_encoder_attention_mask size")
    print(ttm_encoder_attention_mask.shape())

    # layer_head_mask_input_tensor has size [6] and is equal to number of encoder_attention_heads
    # Stays torch tensor and then is used in attention mechanism approprietly for now
    # Because can't convert 1d tensor of size [6] to ttm
    # same for cross_attn_layer_head_mask

    tt_whisper_decoder_layer = TtWhisperDecoderLayer(
        base_address = base_address,
        state_dict = state_dict,
        device = device,
        embed_dim = model.config.d_model,
        num_heads = model.config.decoder_attention_heads,
        decoder_ffn_dim = model.config.decoder_ffn_dim,
        config = model.config,
    )
    tt_whisper_decoder_layer.eval()

    print("********** Decoder layer input shapes **********")
    print(ttm_tensor_hidden_state.shape())
    print(ttm_encoder_hidden_states.shape())

    with torch.no_grad():
        if test_with_all_inputs:
            ttm_output = tt_whisper_decoder_layer(
            hidden_states = ttm_tensor_hidden_state,
            attention_mask = ttm_tensor_attention_mask,
            encoder_hidden_states = ttm_encoder_hidden_states,
            encoder_attention_mask = ttm_encoder_attention_mask,
            layer_head_mask = layer_head_mask_input_tensor,
            cross_attn_layer_head_mask = cross_attn_layer_head_mask,
            past_key_value = None,
            output_attentions = False,
            use_cache=configuration.use_cache,
            )
        else:
            ttm_output = tt_whisper_decoder_layer(
                hidden_states = ttm_tensor_hidden_state,
                encoder_hidden_states = ttm_encoder_hidden_states,
                output_attentions = False,
                use_cache=configuration.use_cache,
            )

        print(ttm_output)

    # Compare results

    ttm_output_to_torch_0 = tt2torch_tensor(ttm_output[0])
    ttm_output_to_torch_0 = torch.squeeze(ttm_output_to_torch_0, 0)

    does_pass, pcc_message = comp_pcc(pytorch_output[0], ttm_output_to_torch_0, 0.98)

    print(comp_allclose(pytorch_output[0], ttm_output_to_torch_0))
    print(pcc_message)

    if does_pass:
        logger.info("Decoder layer Passed!")
    else:
        logger.warning("Decoder layer Failed!")


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_whisper_decoder_layer()
    ttm.device.CloseDevice(device)
