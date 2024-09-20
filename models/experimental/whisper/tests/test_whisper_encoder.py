# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from datasets import load_dataset
from transformers import (
    WhisperModel,
    WhisperForAudioClassification,
    AutoFeatureExtractor,
)

import ttnn
import pytest
from models.utility_functions import is_wormhole_b0, is_blackhole

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from models.experimental.whisper.tt.whisper_encoder import (
    TtWhisperEncoder,
    TtWhisperEncoderOutput,
)


def run_whisper_encoder(device, for_audio_classification=False, encoder_layers=1):
    if for_audio_classification:
        model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    else:
        model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")

    configuration = model.config
    if encoder_layers != configuration.encoder_layers:
        configuration.encoder_layers = encoder_layers
        model = WhisperForAudioClassification(configuration)

    pytorch_model = model.encoder
    pytorch_model.eval()

    base_address = "encoder"
    state_dict = model.state_dict()

    use_real_inputs = True

    if use_real_inputs:
        ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        sample = next(iter(ds))
        inputs = feature_extractor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
        )
        input_features = inputs.input_features
        logger.debug(f"Input of size {input_features.size()}")
        logger.debug("Input audio language:")
        logger.debug(sample["language"])
        head_mask = None
    else:
        batch = 1
        feature_size = 80
        seq_len = 3000
        input_features = torch.rand((batch, feature_size, seq_len))
        head_mask = None

    with torch.no_grad():
        pytorch_output = pytorch_model(
            input_features=input_features,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        tt_whisper_encoder = TtWhisperEncoder(
            base_address=base_address,
            state_dict=state_dict,
            device=device,
            config=pytorch_model.config,
        )
        tt_whisper_encoder.eval()

        input_features = torch2tt_tensor(input_features, device, ttnn.ROW_MAJOR_LAYOUT)
        ttm_output = tt_whisper_encoder(
            input_features=input_features,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        logger.debug(f"Encoder returned {ttm_output.last_hidden_state.get_legacy_shape()}")

        # TT Output To Torch
        ttm_output_pt = tt2torch_tensor(ttm_output.last_hidden_state)
        ttm_output_pt = torch.squeeze(ttm_output_pt, 0)

        logger.debug(f"Encoder output to torch {ttm_output_pt.size()}")

        does_pass, pcc_message = comp_pcc(pytorch_output.last_hidden_state, ttm_output_pt, 0.98)
        logger.info(pcc_message)

        if does_pass:
            logger.info("Encoder Passed!")
        else:
            logger.warning("Encoder Failed!")

        assert does_pass


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_WhipserEncoder_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder(device=device, for_audio_classification=False)


@pytest.mark.skip(reason="Not tested")
def test_WhipserEncoderForAudioClassification_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=24)


@pytest.mark.skip(reason="Not tested")
def test_WhipserEncoderForAudioClassification_one_layer_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=1)


@pytest.mark.skip(reason="Not tested")
def test_WhipserEncoderForAudioClassification_two_layers_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=2)


@pytest.mark.skip(reason="Not tested")
def test_WhipserEncoderForAudioClassification_three_layers_inference(device):
    torch.manual_seed(1234)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=3)
