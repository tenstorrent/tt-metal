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

from transformers import WhisperModel

from utility_functions import pad_weight, tilize_to_list
from python_api_testing.models.whisper.whisper_common import np_compare_tensors, print_corr_coef, torch2tt_tensor, tt2torch_tensor

from libs import tt_lib as ttm

from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as Ttsoftmax
from python_api_testing.models.whisper.whisper_attention import TtWhisperAttention

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

class PytorchWhisperAttention(nn.Module):
    def __init__(self, config, hf_reference_module):
        super().__init__()

        self.attention = hf_reference_module

        # Disable dropout
        self.attention.eval()

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions
    ):
        result = self.attention(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            layer_head_mask = layer_head_mask,
            output_attentions = output_attentions
        )
        return result


def run_whisper_attention(decoder, layer, device):

    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")

    torch.manual_seed(0)
    state_dict = model.state_dict()
    configuration = model.config

    IND = layer
    DECODER = decoder

    # Module to test
    if DECODER:
        print("Decoder attention")
        base_address = f"decoder.layers.{IND}.self_attn"
        hf_reference_module = model.decoder.layers[IND].self_attn
    else:
        print("Encoder attention")
        base_address = f"encoder.layers.{IND}.self_attn"
        hf_reference_module = model.encoder.layers[IND].self_attn

    pytorch_model = PytorchWhisperAttention(configuration, hf_reference_module)

    # Setup input and input parameters
    embd_dim = hf_reference_module.embed_dim
    num_heads = hf_reference_module.num_heads
    batch = 32
    sequence_length = 32
    tgt_len = 32
    src_len = 32

    hidden_state_input_tensor = torch.randn(sequence_length, batch, embd_dim)
    attention_mask_input_tensor = torch.randn(batch, 1, tgt_len, src_len)
    layer_head_mask_input_tensor = torch.randn(num_heads, )

    attn_output, attn_weights_reshaped, past_key_value = pytorch_model(
        hidden_states = hidden_state_input_tensor,
        attention_mask = attention_mask_input_tensor,
        layer_head_mask = layer_head_mask_input_tensor,
        output_attentions = True
    )

    # Convert Inputs to Ttm:
    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)
    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)

    tt_whisper_attention_model = TtWhisperAttention(
        base_address = base_address,
        state_dict = state_dict,
        device = device,
        embed_dim = embd_dim,
        num_heads = num_heads,
        is_decoder = DECODER,
    )

    tt_attn_output, tt_attn_weights_reshaped, tt_past_key_value = tt_whisper_attention_model(
        hidden_states = ttm_tensor_hidden_state,
        attention_mask = ttm_tensor_attention_mask,
        layer_head_mask = layer_head_mask_input_tensor,
        output_attentions = True
    )

    # Check correlation for all output tensors

    # First check: attention output
    tt_attention_to_torch = tt2torch_tensor(tt_attn_output)
    tt_attention_to_torch = tt_attention_to_torch.squeeze(0)
    does_pass, pcc_message = comp_pcc(attn_output, tt_attention_to_torch, 0.98)

    print(comp_allclose(attn_output, tt_attention_to_torch))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Attention output Passed!")
    else:
        logger.warning("Attention output Failed!")

    # Second check: attention weights
    tt_attn_weights_to_torch = tt2torch_tensor(tt_attn_weights_reshaped)
    does_pass, pcc_message = comp_pcc(attn_weights_reshaped, tt_attn_weights_to_torch, 0.98)

    print(comp_allclose(attn_weights_reshaped, tt_attn_weights_to_torch))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test attention output weights Passed!")
    else:
        logger.warning("Test attention output weights Failed!")

    if DECODER:
        tt_past_key_value_to_torch = tt2torch_tensor(tt_past_key_value[0])
        does_pass, pcc_message = comp_pcc(past_key_value[0], tt_past_key_value_to_torch, 0.98)
        print(comp_allclose(past_key_value[0], tt_past_key_value_to_torch))
        print(pcc_message)

        assert does_pass

        if does_pass:
            logger.info("Test attention output weights Passed!")
        else:
            logger.warning("Test attention output weights Failed!")

        tt_past_key_value_to_torch = tt2torch_tensor(tt_past_key_value[1])
        does_pass, pcc_message = comp_pcc(past_key_value[1], tt_past_key_value_to_torch, 0.98)

        print(comp_allclose(past_key_value[1], tt_past_key_value_to_torch))
        print(pcc_message)

        assert does_pass

        if does_pass:
            logger.info("Test attention output weights Passed!")
        else:
            logger.warning("Test attention output weights Failed!")


def test_whipser_encoder_attention():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_attention(decoder=False, layer=0, device=device)
    ttm.device.CloseDevice(device)


def test_whipser_decoder_attention():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_attention(decoder=True, layer=0, device=device)
    ttm.device.CloseDevice(device)
