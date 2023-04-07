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
from python_api_testing.models.whisper.whisper_decoder_layer import TtWhisperDecoderLayer

from libs import tt_lib as ttm

from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from fused_ops.linear import Linear as TtLinear

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_decoder_layer(layer, device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    state_dict = model.state_dict()

    # Accessing the model configuration
    configuration = model.config

    embed_dim = configuration.d_model # 384
    decoder_ffn_dim = configuration.decoder_ffn_dim # 1536
    num_heads = configuration.decoder_attention_heads # 6

    DECODER_IND = layer

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
    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)
    ttm_encoder_hidden_states = torch2tt_tensor(encoder_hidden_states, device)
    ttm_encoder_attention_mask = torch2tt_tensor(encoder_attention_mask, device)

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

    assert does_pass


def test_WhipserDecoderLayer_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_decoder_layer(layer=0, device=device)
    ttm.device.CloseDevice(device)
