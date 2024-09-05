# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from loguru import logger
from datasets import load_dataset
from transformers import WhisperForAudioClassification, AutoFeatureExtractor

import ttnn

from models.experimental.whisper.tt.whisper_for_audio_classification import (
    TtWhisperForAudioClassification,
)
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
    skip_for_wormhole_b0,
)


def run_whisper_for_audio_classification(device):
    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    model.eval()
    state_dict = model.state_dict()
    logger.debug(model.config)

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(ds))

    inputs = feature_extractor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    )

    input_features = inputs.input_features

    logger.debug(f"Input of size {input_features.size()}")  # 1, 80, 3000
    logger.debug("Input audio language:")
    logger.debug(sample["language"])

    with torch.no_grad():
        logits = model(input_features).logits
        logger.debug(logits.size())

    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]

    logger.debug(f"Torch predicted label: {predicted_label}")

    tt_whisper_model = TtWhisperForAudioClassification(state_dict=state_dict, device=device, config=model.config)

    tt_whisper_model.eval()

    with torch.no_grad():
        input_features = torch2tt_tensor(input_features, device, ttnn.ROW_MAJOR_LAYOUT)
        ttm_logits = tt_whisper_model(
            input_features=input_features,
        ).logits

        # Convert to Torch
        ttm_logits = tt2torch_tensor(ttm_logits)
        tt_predicted_class_ids = torch.argmax(ttm_logits).item()
        tt_predicted_label = model.config.id2label[tt_predicted_class_ids]

        logger.info(f"TT predicted label: {tt_predicted_label}")
        logger.info(f"Torch predicted label: {predicted_label}")

        ttm_logits = torch.squeeze(ttm_logits, 0)
        ttm_logits = torch.squeeze(ttm_logits, 0)

        does_pass, pcc_message = comp_pcc(logits, ttm_logits, 0.98)
        logger.info(pcc_message)

        if does_pass:
            logger.info("WhisperForAudioClassification output Passed!")
        else:
            logger.warning("WhisperForAudioClassification output Failed!")

        assert does_pass


@skip_for_wormhole_b0()
def test_WhipserForAudioClassification_inference(device):
    torch.manual_seed(1234)
    run_whisper_for_audio_classification(device=device)
