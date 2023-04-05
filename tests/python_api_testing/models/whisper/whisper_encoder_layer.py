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
from python_api_testing.fused_ops.linear import Linear as TtLinear

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

class TtWhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        embed_dim,
        num_heads,
        encoder_ffn_dim,
        config: WhisperConfig=None
    ):
        super().__init__()

        self.device = device
        self.config = config
        self.state_dict = state_dict

        self.embed_dim = embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim


        self.self_attn = TtWhisperAttention(
            base_address = f"{base_address}.self_attn",
            state_dict = self.state_dict,
            device=self.device,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
        )

        gamma = torch2tt_tensor(self.state_dict[f"{base_address}.self_attn_layer_norm.weight"], ttm.device.GetHost())
        beta = torch2tt_tensor(self.state_dict[f"{base_address}.self_attn_layer_norm.bias"], ttm.device.GetHost())
        tt_gamma = gamma.data()
        tt_beta = beta.data()

        self.self_attn_layer_norm = TtLayernorm(tt_gamma, tt_beta, 1e-05, self.embed_dim, self.embed_dim, device, 1)

        # DO not use DROPOUT for now
        # self.dropout = config.dropout
        # self.activation_dropout = config.activation_dropout

        fc1_weight = torch2tt_tensor(self.state_dict[f"{base_address}.fc1.weight"], ttm.device.GetHost())
        fc1_bias = torch2tt_tensor(self.state_dict[f"{base_address}.fc1.bias"], ttm.device.GetHost())
        fc2_weight = torch2tt_tensor(self.state_dict[f"{base_address}.fc2.weight"], ttm.device.GetHost())
        fc2_bias = torch2tt_tensor(self.state_dict[f"{base_address}.fc2.bias"], ttm.device.GetHost())

        self.fc1 = TtLinear(in_features=self.embed_dim, out_features=self.encoder_ffn_dim, weight=fc1_weight.data(), bias=fc1_bias.data(), device=device)
        self.fc2 = TtLinear(in_features=self.encoder_ffn_dim, out_features=self.embed_dim, weight=fc2_weight.data(), bias=fc2_bias.data(), device=device)

        gamma_1 = torch2tt_tensor(self.state_dict[f"{base_address}.final_layer_norm.weight"], ttm.device.GetHost())
        beta_1 = torch2tt_tensor(self.state_dict[f"{base_address}.final_layer_norm.bias"], ttm.device.GetHost())
        tt_gamma_1 = gamma_1.data()
        tt_beta_1 = beta_1.data()

        self.final_layer_norm = TtLayernorm(tt_gamma_1, tt_beta_1, 1e-05, self.embed_dim, self.embed_dim, device, 1)


    def forward(
        self,
        hidden_states: ttm.tensor.Tensor,
        attention_mask: ttm.tensor.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> ttm.tensor.Tensor:
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

        H_hidden_states = hidden_states.shape()[-2]

        hidden_states = self.self_attn_layer_norm(hidden_states,overrideH=H_hidden_states)

        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # Do not use dropout for now
        #hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = ttm.tensor.add(hidden_states, residual)

        residual = hidden_states

        H_hidden_states = hidden_states.shape()[-2]
        hidden_states = self.final_layer_norm(hidden_states, overrideH=H_hidden_states)

        hidden_states = self.fc1(hidden_states)

        hidden_states = ttm.tensor.gelu(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = ttm.tensor.add(hidden_states, residual)
        hidden_states_torch = tt2torch_tensor(hidden_states)

        if hidden_states_torch.dtype == torch.float16 and (
            torch.isinf(hidden_states_torch).any() or torch.isnan(hidden_states_torch).any()
        ):
            clamp_value = torch.finfo(hidden_states_torch.dtype).max - 1000
            hidden_states_torch = torch.clamp(hidden_states_torch, min=-clamp_value, max=clamp_value)

        hidden_states = torch2tt_tensor(hidden_states_torch, self.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


def run_whisper_encoder_layer():
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    state_dict = model.state_dict()
    # Accessing the model configuration
    configuration = model.config
    embed_dim = configuration.d_model # 384
    encoder_ffn_dim = configuration.encoder_ffn_dim # 1536
    num_heads = configuration.encoder_attention_heads # 6

    """ Huggingface Whisper Encoder Layer """
    ENCODER_IND = 0
    pytorch_model = model.encoder.layers[ENCODER_IND]
    pytorch_model.eval()
    base_address = f"encoder.layers.{ENCODER_IND}"

    batch = 32
    tgt_len = 32
    src_len = 32

    hidden_state_input_tensor = torch.randn(src_len, batch, embed_dim)
    attention_mask_input_tensor = torch.randn(batch, 1, tgt_len, src_len)
    layer_head_mask_input_tensor = torch.randn(num_heads, )

    print("Sizes of input tensors for encoder layer")
    print(hidden_state_input_tensor.size())
    print(attention_mask_input_tensor.size())
    print(layer_head_mask_input_tensor.size())

    pytorch_output = pytorch_model(
        hidden_state_input_tensor,
        attention_mask_input_tensor,
        layer_head_mask_input_tensor
    )

    """ TTM Whisper Encoder Layer """

    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)
    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)

    # layer_head_mask_input_tensor has size [6] and is equal to number of encoder_attention_heads
    # Stays torch tensor and then is used in attention mechanism approprietly for now
    # Because can't convert 1d tensor of size [6] to ttm

    tt_whisper_encoder_layer = TtWhisperEncoderLayer(
        base_address = base_address,
        state_dict = state_dict,
        device = device,
        embed_dim = embed_dim,
        num_heads = num_heads,
        encoder_ffn_dim = encoder_ffn_dim
    )
    ttm_output = tt_whisper_encoder_layer(
        ttm_tensor_hidden_state,
        ttm_tensor_attention_mask,
        layer_head_mask_input_tensor,
        True
    )

    print(ttm_output) # Returns Tuple of size 2
    # Attention weights Not used
    # Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
    # but it is not used

    debug = False
    if debug:
        print("Hidden states")
        ttm_output[0].pretty_print()

    ttm_output_to_torch_0 = tt2torch_tensor(ttm_output[0])
    ttm_output_to_torch_0 = torch.squeeze(ttm_output_to_torch_0, 0)
    does_pass, pcc_message = comp_pcc(pytorch_output[0], ttm_output_to_torch_0, 0.98)

    print(comp_allclose(pytorch_output[0], ttm_output_to_torch_0))
    print(pcc_message)

    if does_pass:
        logger.info("Encoder layer Passed!")
    else:
        logger.warning("Encoder layer Failed!")


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_whisper_encoder_layer()
    ttm.device.CloseDevice(device)
