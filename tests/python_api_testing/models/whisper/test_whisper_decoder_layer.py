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

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor, create_padded_tensor, create_unpadded_tensor
from python_api_testing.models.whisper.whisper_decoder_layer import TtWhisperDecoderLayer

from libs import tt_lib as ttm

from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from fused_ops.linear import Linear as TtLinear

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_decoder_layer(layer, device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    state_dict = model.state_dict()

    configuration = model.config

    embed_dim = configuration.d_model
    decoder_ffn_dim = configuration.decoder_ffn_dim
    num_heads = configuration.decoder_attention_heads

    DECODER_IND = layer

    pytorch_model = model.decoder.layers[DECODER_IND]
    pytorch_model.eval()

    base_address = f"decoder.layers.{DECODER_IND}"

    padding = True
    if padding:
        enc_seq_len = 1500
        pad = 4
    else:
        enc_seq_len = 32

    batch = 1
    tgt_len = 32
    seq_len = 32

    # Similary to what Decoder's method self._prepare_decoder_attention_mask returns
    attention_mask_input_tensor = (torch.rand(size=(1,1,tgt_len,seq_len)) < 0.25).int().float() * -3.4028e+38

    hidden_state_input_tensor = torch.rand(batch, seq_len, embed_dim)
    encoder_hidden_states = torch.rand(batch, enc_seq_len, embed_dim)
    encoder_attention_mask = None #torch.ones(batch, 1, tgt_len, enc_seq_len)

    layer_head_mask_input_tensor = torch.rand(num_heads, )
    cross_attn_layer_head_mask = torch.rand(num_heads)
    past_key_value = None

    test_with_all_inputs = False

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
                attention_mask = attention_mask_input_tensor,
                output_attentions = False,
                use_cache=configuration.use_cache,
            )

    """ TTM Whisper Decoder Layer """

    # Make inputs ready convert to tt/pad
    if padding:

        output_tensor_shape = list(encoder_hidden_states.size())
        while len(output_tensor_shape) < 4:
            output_tensor_shape.insert(0,1)
        output_tensor_shape[-2] = enc_seq_len + pad
        ttm_encoder_hidden_states = create_padded_tensor(list(encoder_hidden_states.size()), encoder_hidden_states, output_tensor_shape, pad_value=0.0, device=device)

        if encoder_attention_mask:
            output_tensor_shape = list(encoder_attention_mask.size())
            while len(output_tensor_shape) < 4:
                output_tensor_shape.insert(0,1)
            output_tensor_shape[-1] = enc_seq_len + pad

            ttm_encoder_attention_mask = create_padded_tensor(list(encoder_attention_mask.size()), encoder_hidden_states, output_tensor_shape, pad_value=0.0, device=device)
        else:
            ttm_encoder_attention_mask = None

    else:
        ttm_encoder_hidden_states = torch2tt_tensor(encoder_hidden_states, device)
        if encoder_attention_mask:
            ttm_encoder_attention_mask = torch2tt_tensor(encoder_attention_mask, device)
        else:
            ttm_encoder_attention_mask = None

    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)
    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)

    # TODO: Support this parameter as tt tensor with padding
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

    logger.info("Running Tt Whisper Decoder layer")

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
                attention_mask = ttm_tensor_attention_mask,
                encoder_hidden_states = ttm_encoder_hidden_states,
                output_attentions = False,
                use_cache=configuration.use_cache,
            )

    logger.info("Decoder layer finished. Compare results")

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

if __name__ == "__main__":
    test_WhipserDecoderLayer_inference()
