# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import WhisperModel, WhisperConfig

import ttnn

from models.experimental.whisper.tt.whisper_decoder_layer import (
    TtWhisperDecoderLayer,
)
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)


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

    enc_seq_len = 1500
    batch = 1
    tgt_len = 32
    seq_len = 32

    # Similary to what Decoder's method self._prepare_decoder_attention_mask returns
    attention_mask_input_tensor = (torch.rand(size=(1, 1, tgt_len, seq_len)) < 0.25).int().float() * -3.4028e38

    hidden_state_input_tensor = torch.rand(batch, seq_len, embed_dim)
    encoder_hidden_states = torch.rand(batch, enc_seq_len, embed_dim)
    encoder_attention_mask = None

    layer_head_mask_input_tensor = torch.rand(
        num_heads,
    )
    cross_attn_layer_head_mask = torch.rand(
        num_heads,
    )
    past_key_value = None

    test_with_all_inputs = True

    with torch.no_grad():
        if test_with_all_inputs:
            pytorch_output = pytorch_model(
                hidden_states=hidden_state_input_tensor,
                attention_mask=attention_mask_input_tensor,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask_input_tensor,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=False,
                use_cache=configuration.use_cache,
            )
        else:
            pytorch_output = pytorch_model(
                hidden_states=hidden_state_input_tensor,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask_input_tensor,
                output_attentions=False,
                use_cache=configuration.use_cache,
            )

    """ TTM Whisper Decoder Layer """
    ttm_encoder_hidden_states = torch2tt_tensor(encoder_hidden_states, device, ttnn.ROW_MAJOR_LAYOUT)

    if encoder_attention_mask:
        ttm_encoder_attention_mask = torch2tt_tensor(encoder_attention_mask, device, ttnn.ROW_MAJOR_LAYOUT)
    else:
        ttm_encoder_attention_mask = None

    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device, ttnn.ROW_MAJOR_LAYOUT)
    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device, ttnn.ROW_MAJOR_LAYOUT)

    # TODO: Support this parameter as tt tensor with padding
    # layer_head_mask_input_tensor has size [6] and is equal to number of encoder_attention_heads
    # Stays torch tensor and then is used in attention mechanism approprietly for now
    # Because can't convert 1d tensor of size [6] to tt_lib
    # same for cross_attn_layer_head_mask

    layer_head_mask_input_tensor = layer_head_mask_input_tensor.view(1, 1, 1, num_heads)
    layer_head_mask_input_tensor = torch2tt_tensor(layer_head_mask_input_tensor, device, ttnn.ROW_MAJOR_LAYOUT)
    cross_attn_layer_head_mask = cross_attn_layer_head_mask.view(1, 1, 1, num_heads)
    cross_attn_layer_head_mask = torch2tt_tensor(cross_attn_layer_head_mask, device, ttnn.ROW_MAJOR_LAYOUT)

    tt_whisper_decoder_layer = TtWhisperDecoderLayer(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        embed_dim=model.config.d_model,
        num_heads=model.config.decoder_attention_heads,
        decoder_ffn_dim=model.config.decoder_ffn_dim,
        config=model.config,
    )
    tt_whisper_decoder_layer.eval()

    logger.info("Running Tt Whisper Decoder layer")

    with torch.no_grad():
        if test_with_all_inputs:
            ttm_output = tt_whisper_decoder_layer(
                hidden_states=ttm_tensor_hidden_state,
                attention_mask=ttm_tensor_attention_mask,
                encoder_hidden_states=ttm_encoder_hidden_states,
                encoder_attention_mask=ttm_encoder_attention_mask,
                layer_head_mask=layer_head_mask_input_tensor,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=False,
                use_cache=configuration.use_cache,
            )
        else:
            ttm_output = tt_whisper_decoder_layer(
                hidden_states=ttm_tensor_hidden_state,
                attention_mask=ttm_tensor_attention_mask,
                encoder_hidden_states=ttm_encoder_hidden_states,
                output_attentions=False,
                use_cache=configuration.use_cache,
            )

    logger.info("Decoder layer finished. Compare results")

    ttm_output_to_torch_0 = tt2torch_tensor(ttm_output[0])
    ttm_output_to_torch_0 = torch.squeeze(ttm_output_to_torch_0, 0)

    does_pass, pcc_message = comp_pcc(pytorch_output[0], ttm_output_to_torch_0, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Decoder layer Passed!")
    else:
        logger.warning("Decoder layer Failed!")

    assert does_pass


def test_WhipserDecoderLayer_inference(device):
    torch.manual_seed(1234)
    run_whisper_decoder_layer(layer=0, device=device)
