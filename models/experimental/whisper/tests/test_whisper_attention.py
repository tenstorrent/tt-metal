# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from loguru import logger
from transformers import WhisperModel, WhisperForAudioClassification

import ttnn

from models.experimental.whisper.tt.whisper_attention import TtWhisperAttention
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, comp_pcc


class PytorchWhisperAttention(nn.Module):
    def __init__(self, hf_reference_module):
        super().__init__()
        self.attention = hf_reference_module
        # Disable dropout
        self.attention.eval()

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        result = self.attention(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        return result


def run_whisper_attention(decoder, layer, device, for_audio_classification, is_self_attn=True):
    if for_audio_classification:
        model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        logger.info("Using WhisperForAudioClassification model")
    else:
        model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
        logger.info("Using Whisper tiny model")

    state_dict = model.state_dict()
    configuration = model.config
    logger.info(f"output attentions: {configuration.output_attentions}")

    IND = layer
    DECODER = decoder
    USE_TORCH_SOFTMAX = True
    BATCH = 1500

    # Module to test
    if DECODER:
        if is_self_attn:
            base_address = f"decoder.layers.{IND}.self_attn"
            hf_reference_module = model.decoder.layers[IND].self_attn
            logger.info("Decoder Self attention")
        else:
            base_address = f"decoder.layers.{IND}.encoder_attn"
            hf_reference_module = model.decoder.layers[IND].encoder_attn
            logger.info("Decoder Encoder attention")
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

    if for_audio_classification or not DECODER:
        # Encoder inputs
        logger.info("Making inputs ready for encoder")
        hidden_state_input_tensor = torch.rand(1, BATCH, embd_dim)
        ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        # Decoder inputs
        hidden_state_input_tensor = torch.rand(1, 32, embd_dim)
        ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        if not is_self_attn:
            key_value_states = torch.rand(1, BATCH, embd_dim)
            ttm_tensor_key_value_states = torch2tt_tensor(key_value_states, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

    if decoder and is_self_attn:
        # Decoder self attention
        attention_mask_input_tensor = (torch.rand(size=(1, 1, 32, 32)) < 0.25).int().float() * -3.4028e38
        ttm_tensor_attention_mask = torch2tt_tensor(
            attention_mask_input_tensor,
            device,
            tt_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    else:
        # Decoder encoder attention
        attention_mask_input_tensor = None
        ttm_tensor_attention_mask = None

    logger.info(f"Running torch whisper attention")
    with torch.no_grad():
        attn_output, attn_weights_reshaped, past_key_value = pytorch_model(
            hidden_states=hidden_state_input_tensor,
            key_value_states=key_value_states,
            attention_mask=attention_mask_input_tensor,
            output_attentions=configuration.output_attentions,
        )

    logger.info(f"Making tt whisper attention object")

    tt_whisper_attention_model = TtWhisperAttention(
        config=model.config,
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        embed_dim=embd_dim,
        num_heads=num_heads,
        is_decoder=DECODER,
    )

    logger.info(f"Running tt whisper attention")

    with torch.no_grad():
        (
            tt_attn_output,
            tt_attn_weights_reshaped,
            tt_past_key_value,
        ) = tt_whisper_attention_model(
            hidden_states=ttm_tensor_hidden_state,
            key_value_states=ttm_key_value_states,
            attention_mask=ttm_tensor_attention_mask,
            output_attentions=configuration.output_attentions,
        )

    logger.info(f"Finish running tt whisper attention")

    tt_attn_output_to_torch = tt2torch_tensor(tt_attn_output)
    tt_attn_output_to_torch = tt_attn_output_to_torch.squeeze(0)

    logger.info(f"Cheking attention ouptuts")

    tt_attention_to_torch = tt2torch_tensor(tt_attn_output)
    tt_attention_to_torch = tt_attention_to_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(attn_output, tt_attention_to_torch, 0.98)
    logger.info(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Attention output Passed!")
    else:
        logger.warning("Attention output Failed!")

    if configuration.output_attentions:
        # Check other outputs from attention
        # Second check: attention weights

        tt_attn_weights_to_torch = tt2torch_tensor(tt_attn_weights_reshaped)
        logger.debug(attn_weights_reshaped.size())
        logger.debug(tt_attn_weights_to_torch.size())

        does_pass, pcc_message = comp_pcc(attn_weights_reshaped, tt_attn_weights_to_torch, 0.98)
        logger.info(pcc_message)

        assert does_pass

        if does_pass:
            logger.info("Test attention output weights Passed!")
        else:
            logger.warning("Test attention output weights Failed!")

        if DECODER:
            tt_past_key_value_to_torch = tt2torch_tensor(tt_past_key_value[0])

            does_pass, pcc_message = comp_pcc(past_key_value[0], tt_past_key_value_to_torch, 0.98)
            logger.info(pcc_message)

            assert does_pass

            if does_pass:
                logger.info("Test attention output weights Passed!")
            else:
                logger.warning("Test attention output weights Failed!")

            tt_past_key_value_to_torch = tt2torch_tensor(tt_past_key_value[1])

            does_pass, pcc_message = comp_pcc(past_key_value[1], tt_past_key_value_to_torch, 0.98)
            logger.info(pcc_message)

            if does_pass:
                logger.info("Test attention output weights Passed!")
            else:
                logger.warning("Test attention output weights Failed!")

            assert does_pass


def test_WhisperEncoderAttention_inference(device):
    torch.manual_seed(1234)

    run_whisper_attention(decoder=False, layer=0, device=device, for_audio_classification=False)


def test_WhisperDecoderEncoderAttention_inference(device):
    torch.manual_seed(1234)

    run_whisper_attention(
        decoder=True,
        layer=0,
        device=device,
        for_audio_classification=False,
        is_self_attn=False,
    )


def test_WhisperDecoderSelfAttention_inference(device):
    torch.manual_seed(1234)

    run_whisper_attention(
        decoder=True,
        layer=0,
        device=device,
        for_audio_classification=False,
        is_self_attn=True,
    )


def test_WhisperEncoderForAudioClassificationAttention_inference(device):
    torch.manual_seed(1234)

    run_whisper_attention(decoder=False, layer=0, device=device, for_audio_classification=True)
