# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from typing import Tuple
from loguru import logger
from transformers import WhisperModel, WhisperForAudioClassification


import ttnn
import pytest

from models.experimental.whisper.tt.whisper_encoder_layer import (
    TtWhisperEncoderLayer,
)
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
    is_wormhole_b0,
)

pytestmark = pytest.mark.skipif(is_wormhole_b0(), reason="Skip for Wormhole B0")


def run_whisper_encoder_layer(layer, device, for_audio_classification=False):
    if for_audio_classification:
        model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    else:
        model = WhisperModel.from_pretrained("openai/whisper-tiny.en")

    state_dict = model.state_dict()
    configuration = model.config

    embed_dim = configuration.d_model
    encoder_ffn_dim = configuration.encoder_ffn_dim
    num_heads = configuration.encoder_attention_heads

    """ Huggingface Whisper Encoder Layer """
    ENCODER_IND = layer
    pytorch_model = model.encoder.layers[ENCODER_IND]
    pytorch_model.eval()
    base_address = f"encoder.layers.{ENCODER_IND}"

    hidden_state_input_tensor = torch.rand(1, 1500, embed_dim)
    attention_mask_input_tensor = None
    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device, ttnn.ROW_MAJOR_LAYOUT)
    ttm_tensor_attention_mask = None
    layer_head_mask_input_tensor = None

    with torch.no_grad():
        pytorch_output = pytorch_model(
            hidden_state_input_tensor,
            attention_mask_input_tensor,
            layer_head_mask_input_tensor,
        )

    """ TTM Whisper Encoder Layer """

    # layer_head_mask_input_tensor has size [6] and is equal to number of encoder_attention_heads
    # Stays torch tensor and then is used in attention mechanism approprietly for now
    # Because can't convert 1d tensor of size [6] to tt_lib

    tt_whisper_encoder_layer = TtWhisperEncoderLayer(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        config=configuration,
    )
    with torch.no_grad():
        ttm_output = tt_whisper_encoder_layer(
            ttm_tensor_hidden_state,
            ttm_tensor_attention_mask,
            layer_head_mask_input_tensor,
            True,
        )

    # Returns Tuple of size 2
    # Attention weights Not used
    # Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
    # but it is not used
    # First check: attention output

    # Unpad output tensor
    input_tensors_shape = ttm_output[0].get_legacy_shape()
    logger.info(input_tensors_shape)

    ttm_output_to_torch_0 = tt2torch_tensor(ttm_output[0])
    ttm_output_to_torch_0 = ttm_output_to_torch_0.squeeze(0)

    does_pass, pcc_message = comp_pcc(pytorch_output[0], ttm_output_to_torch_0, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Encoder layer Passed!")
    else:
        logger.warning("Encoder layer Failed!")

    assert does_pass


def test_WhipserEncoderLayer_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder_layer(layer=0, device=device)


@pytest.mark.skip(reason="Not tested")
def test_WhisperEncoderLayerForAudioClassification_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder_layer(layer=0, device=device, for_audio_classification=True)
