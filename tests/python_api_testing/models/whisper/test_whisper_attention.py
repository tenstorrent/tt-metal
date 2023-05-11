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

from transformers import WhisperModel, WhisperForAudioClassification

from utility_functions import pad_weight, tilize_to_list
from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor, create_padded_tensor, create_unpadded_tensor

from libs import tt_lib as ttm

from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as Ttsoftmax
from python_api_testing.models.whisper.whisper_attention import TtWhisperAttention

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

class PytorchWhisperAttention(nn.Module):
    def __init__(self, hf_reference_module):
        super().__init__()

        self.attention = hf_reference_module

        # Disable dropout
        self.attention.eval()

    def forward(
        self,
        hidden_states,
        key_value_states = None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        result = self.attention(
            hidden_states = hidden_states,
            key_value_states = key_value_states,
            attention_mask = attention_mask,
            layer_head_mask = layer_head_mask,
            output_attentions = output_attentions
        )
        return result


def run_whisper_attention(decoder, layer, device, for_audio_classification, is_self_attn = True):

    if for_audio_classification:
        model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        print("Using WhisperForAudioClassification model")
    else:
        model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
        print("Using base model")

    state_dict = model.state_dict()
    configuration = model.config
    print(configuration.output_attentions)

    IND = layer
    DECODER = decoder
    USE_TORCH_SOFTMAX = True

    padding = True
    if padding:
        BATCH = 1500
        PAD = 4
        logger.info(f"Using padding")

    # Module to test
    if DECODER:
        logger.info("Decoder attention")
        if is_self_attn:
            base_address = f"decoder.layers.{IND}.self_attn"
            hf_reference_module = model.decoder.layers[IND].self_attn
        else:
            base_address = f"decoder.layers.{IND}.encoder_attn"
            hf_reference_module = model.decoder.layers[IND].encoder_attn
    else:
        logger.info("Encoder attention")
        base_address = f"encoder.layers.{IND}.self_attn"
        hf_reference_module = model.encoder.layers[IND].self_attn

    pytorch_model = PytorchWhisperAttention(hf_reference_module)

    # Setup input and input parameters
    embd_dim = hf_reference_module.embed_dim
    num_heads = hf_reference_module.num_heads

    key_value_states = None
    ttm_key_value_states = None

    # Encoder inputs
    if for_audio_classification or not DECODER:
        # Torch inputs
        logger.info("Making inputs ready for encoder")
        hidden_state_input_tensor = torch.rand(1, BATCH, embd_dim)

        # Convert to tt inputs
        if padding:
            output_tensor_shape = list(hidden_state_input_tensor.size())
            output_tensor_shape.insert(0,1)
            output_tensor_shape[-2] = BATCH + PAD
            ttm_tensor_hidden_state = create_padded_tensor(list(hidden_state_input_tensor.size()), hidden_state_input_tensor, output_tensor_shape, pad_value=0.0, device=device)

        else:
            ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)
    # Decoder inputs
    else:
        hidden_state_input_tensor = torch.rand(1, 32, embd_dim)
        ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)

        if not is_self_attn:
            key_value_states = torch.rand(1, BATCH, embd_dim)
            # Convert to tt inputs
            # Pad
            output_tensor_shape = list(key_value_states.size())
            output_tensor_shape.insert(0,1)
            output_tensor_shape[-2] = BATCH + PAD
            ttm_key_value_states = create_padded_tensor(list(key_value_states.size()), key_value_states, output_tensor_shape, pad_value=0.0, device=device)

    if decoder and is_self_attn:
        # Decoder self attention
        attention_mask_input_tensor = (torch.rand(size=(1,1,32,32)) < 0.25).int().float() * -3.4028e+38
        ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)
    else:
        # Decoder encoder attention
        attention_mask_input_tensor = None
        ttm_tensor_attention_mask = None


    logger.info(f"Running torch whisper attention")
    with torch.no_grad():
        attn_output, attn_weights_reshaped, past_key_value = pytorch_model(
            hidden_states = hidden_state_input_tensor,
            key_value_states = key_value_states,
            attention_mask = attention_mask_input_tensor,
            output_attentions = configuration.output_attentions,
        )

    logger.info(f"Running tt whisper attention")

    tt_whisper_attention_model = TtWhisperAttention(
        config = model.config,
        base_address = base_address,
        state_dict = state_dict,
        device = device,
        embed_dim = embd_dim,
        num_heads = num_heads,
        is_decoder = DECODER,
        use_torch_softmax = USE_TORCH_SOFTMAX,
    )

    with torch.no_grad():
        tt_attn_output, tt_attn_weights_reshaped, tt_past_key_value = tt_whisper_attention_model(
            hidden_states = ttm_tensor_hidden_state,
            key_value_states = ttm_key_value_states,
            attention_mask = ttm_tensor_attention_mask,
            output_attentions = configuration.output_attentions,
        )

    tt_attn_output_to_torch = tt2torch_tensor(tt_attn_output)
    tt_attn_output_to_torch = tt_attn_output_to_torch.squeeze(0)

    logger.info(f"Cheking attention ouptuts")

    if is_self_attn and DECODER:
        padding = False

    if for_audio_classification or not DECODER:

        if padding:
            input_tensors_shape = tt_attn_output.shape()
            input_tensors_shape[-2] = 1500
            tt_attn_output = create_unpadded_tensor(tt_attn_output, input_tensors_shape)
            tt_attention_to_torch = torch.Tensor(tt_attn_output.data()).reshape(*tt_attn_output.shape())
        else:
            tt_attention_to_torch = tt2torch_tensor(tt_attn_output)
    else:
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

    if configuration.output_attentions:
        # Check other outputs from attention

        # Second check: attention weights
        if padding:
            input_tensors_shape = tt_attn_weights_reshaped.shape()
            input_tensors_shape[-1] = 1500
            tt_attn_weights_reshaped = create_unpadded_tensor(tt_attn_weights_reshaped, input_tensors_shape)
            tt_attn_weights_to_torch = torch.Tensor(tt_attn_weights_reshaped.data()).reshape(*tt_attn_weights_reshaped.shape())
        else:
            tt_attn_weights_to_torch = tt2torch_tensor(tt_attn_weights_reshaped)
            print(attn_weights_reshaped.size())
            print(tt_attn_weights_to_torch.size())

        does_pass, pcc_message = comp_pcc(attn_weights_reshaped, tt_attn_weights_to_torch, 0.98)

        print(comp_allclose(attn_weights_reshaped, tt_attn_weights_to_torch))
        print(pcc_message)

        assert does_pass

        if does_pass:
            logger.info("Test attention output weights Passed!")
        else:
            logger.warning("Test attention output weights Failed!")

        if DECODER:

            if padding:
                input_tensors_shape = tt_past_key_value[0].shape()
                input_tensors_shape[-2] = BATCH
                tt_past_key_value_0 = create_unpadded_tensor(tt_past_key_value[0], input_tensors_shape)
                tt_past_key_value_to_torch = torch.Tensor(tt_past_key_value_0.data()).reshape(*tt_past_key_value_0.shape())

            else:
                tt_past_key_value_to_torch = tt2torch_tensor(tt_past_key_value[0])

            does_pass, pcc_message = comp_pcc(past_key_value[0], tt_past_key_value_to_torch, 0.98)
            print(comp_allclose(past_key_value[0], tt_past_key_value_to_torch))
            print(pcc_message)

            assert does_pass

            if does_pass:
                logger.info("Test attention output weights Passed!")
            else:
                logger.warning("Test attention output weights Failed!")

            if padding:
                input_tensors_shape = tt_past_key_value[1].shape()
                input_tensors_shape[-2] = BATCH

                tt_past_key_value_1 = create_unpadded_tensor(tt_past_key_value[1], input_tensors_shape)
                tt_past_key_value_to_torch = torch.Tensor(tt_past_key_value_1.data()).reshape(*tt_past_key_value_1.shape())
            else:
                tt_past_key_value_to_torch = tt2torch_tensor(tt_past_key_value[1])

            does_pass, pcc_message = comp_pcc(past_key_value[1], tt_past_key_value_to_torch, 0.98)

            print(comp_allclose(past_key_value[1], tt_past_key_value_to_torch))
            print(pcc_message)

            if does_pass:
                logger.info("Test attention output weights Passed!")
            else:
                logger.warning("Test attention output weights Failed!")

            assert does_pass

def test_WhisperEncoderAttention_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_attention(decoder=False, layer=0, device=device, for_audio_classification=False)
    ttm.device.CloseDevice(device)

def test_WhisperDecoderEncoderAttention_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_attention(decoder=True, layer=0, device=device, for_audio_classification=False, is_self_attn = False)
    ttm.device.CloseDevice(device)

def test_WhisperDecoderSelfAttention_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_attention(decoder=True, layer=0, device=device, for_audio_classification=False, is_self_attn = True)
    ttm.device.CloseDevice(device)

def test_WhisperEncoderForAudioClassificationAttention_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_attention(decoder=False, layer=0, device=device, for_audio_classification=True)
    ttm.device.CloseDevice(device)

if __name__ =="__main__":
    test_WhisperEncoderAttention_inference()
    test_WhisperDecoderEncoderAttention_inference()
    test_WhisperDecoderSelfAttention_inference()
    test_WhisperEncoderForAudioClassificationAttention_inference()
