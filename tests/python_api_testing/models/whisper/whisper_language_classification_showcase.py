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

    # change config
    configuration = model.config
    configuration.max_source_positions = 1024

    model = WhisperForAudioClassification(configuration)

    model.eval()
    state_dict = model.state_dict()

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(ds))

    inputs = feature_extractor(
        sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
    )
    # Take only 2048 features on last dim. 3000 not supported because of shape and encoder max_source_positions
    input_features = inputs.input_features
    input_features = input_features[:,:,:2048]

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

        print("Sample info")
        print(sample)

        print("Sample ID")
        print(sample["id"])

        print("Loading Input Sample audio transcription...")
        print(sample["transcription"])

        tt_predicted_class_ids = torch.argmax(ttm_logits).item()
        tt_predicted_label = model.config.id2label[tt_predicted_class_ids]
        print(f"TT predicted label: {tt_predicted_label}")

        if tt_predicted_label == predicted_label:
            does_pass = True
            print("Predicted classes match")
        else:
            does_pass = False
            print("Predicted classes don't match")

        if does_pass:
            logger.info("WhisperForAudioClassification output Passed!")
        else:
            logger.warning("WhisperForAudioClassification output Failed!")

        assert does_pass


if __name__=="__main__":
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_for_audio_classification(device=device)
    ttm.device.CloseDevice(device)
