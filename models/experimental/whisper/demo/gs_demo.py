# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import load_dataset
from loguru import logger

import ttnn
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.whisper.tt.whisper_for_audio_classification import (
    TtWhisperForAudioClassification,
)


def test_gs_demo():
    torch.manual_seed(1234)
    device = ttnn.CreateDevice(0)

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

    with open("sample_audio.npy", "wb") as f:
        np.save(f, sample["audio"]["array"])
    # # actually save the input
    logger.info(f"Input audio is saved as sample_audio.npy.")
    logger.info(f"GS's predicted Output: {tt_predicted_label}.")

    ttnn.CloseDevice(device)
