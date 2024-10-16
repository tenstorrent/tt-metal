# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import load_dataset
import numpy as np
import torch
from loguru import logger


def test_cpu_demo():
    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(ds))

    inputs = feature_extractor(
        sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
    )
    input_features = inputs.input_features

    with torch.no_grad():
        logits = model(input_features).logits

    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]

    with open("sample_audio.npy", "wb") as f:
        np.save(f, sample["audio"]["array"])
    # actually save the input
    logger.info(f"Input audio is saved as sample_audio.npy.")
    logger.info(f"CPU's predicted Output: {predicted_label}.")
