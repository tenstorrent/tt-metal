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

from transformers import WhisperProcessor, WhisperForAudioClassification, AutoFeatureExtractor, AutoProcessor, WhisperConfig
from datasets import load_dataset

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.whisper.whisper_for_audio_classification import TtWhisperForAudioClassification, TtWhisperForAudioClassificationOutput

from libs import tt_lib as ttm

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

def run_whisper_for_audio_classification(device):
    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    model.eval()
    state_dict = model.state_dict()
    print(model.config)

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(ds))

    inputs = feature_extractor(
        sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
    )

    input_features = inputs.input_features
    print(f"Input of size {input_features.size()}") # 1, 80, 3000
    print("Input audio language:")
    print(sample["language"])

    with torch.no_grad():
        logits = model(input_features).logits
        print(logits.size())

    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    print(f"Torch predicted label: {predicted_label}")

    tt_whisper_model = TtWhisperForAudioClassification(
        state_dict=state_dict,
        device=device,
        config=model.config
    )
    tt_whisper_model.eval()

    with torch.no_grad():

        ttm_logits = tt_whisper_model(
            input_features = input_features,
        ).logits
        print(ttm_logits)
        # Convert to Torch
        ttm_logits = torch.Tensor(ttm_logits.data()).reshape(ttm_logits.shape())
        tt_predicted_class_ids = torch.argmax(ttm_logits).item()
        tt_predicted_label = model.config.id2label[tt_predicted_class_ids]

        print(f"TT predicted label: {tt_predicted_label}")
        print(f"Torch predicted label: {predicted_label}")

        ttm_logits = torch.squeeze(ttm_logits, 0)
        ttm_logits = torch.squeeze(ttm_logits, 0)

        does_pass, pcc_message = comp_pcc(logits, ttm_logits, 0.98)

        print(comp_allclose(logits, ttm_logits))
        print(pcc_message)

        if does_pass:
            logger.info("WhisperForAudioClassification output Passed!")
        else:
            logger.warning("WhisperForAudioClassification output Failed!")

        assert does_pass


def test_WhipserForAudioClassification_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_for_audio_classification(device=device)
    ttm.device.CloseDevice(device)

if __name__=="__main__":
    test_WhipserForAudioClassification_inference()
