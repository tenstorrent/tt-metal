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
from dataclasses import dataclass
from datasets import load_dataset

import random
from typing import Optional, Tuple, Union
from loguru import logger

from transformers import WhisperModel, WhisperConfig, AutoFeatureExtractor

from libs import tt_lib as ttm
from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.whisper.whisper_model import TtWhisperModel, TtWhisperModelOutput

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_model(device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = model.config

    # Change config and re-initialize model
    configuration.max_source_positions = 1024

    pytorch_model = WhisperModel(configuration)
    pytorch_model.eval()

    state_dict = pytorch_model.state_dict()

    """
    Original example from Huggingface documentation:

    Example:
        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, WhisperModel
        >>> from datasets import load_dataset

        >>> model = WhisperModel.from_pretrained("openai/whisper-base")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 2, 512]
        ```
    """

    # Define inputs
    create_synthetic_inputs = False

    if create_synthetic_inputs:
        batch = 1
        feature_size = 80
        seq_len = 2048 # original from HF example should be: seq_len = 3000, when max_source_positions=1500
        input_features = torch.rand((batch, feature_size, seq_len))
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        # original from HF example should be: seq_len = 3000, when max_source_positions=1500
        input_features = inputs.input_features[:,:,:2048]

    dec_seq_len = 32
    decoder_input_ids = torch.tensor([[1,] * dec_seq_len]) * pytorch_model.config.decoder_start_token_id

    with torch.no_grad():
        pytorch_output = pytorch_model(
            input_features = input_features,
            decoder_input_ids=decoder_input_ids
        )

    print("****** TTM WhisperModel ******")
    tt_whisper = TtWhisperModel(
        state_dict=state_dict,
        device=device,
        config=pytorch_model.config
    )
    tt_whisper.eval()

    with torch.no_grad():
        ttm_output = tt_whisper(
            input_features = input_features,
            decoder_input_ids=decoder_input_ids
        )

    # Check correlations
    tt_out_to_torch = tt2torch_tensor(ttm_output.last_hidden_state)
    tt_out_to_torch = torch.squeeze(tt_out_to_torch, 0)

    does_pass, pcc_message = comp_pcc(pytorch_output.last_hidden_state, tt_out_to_torch, 0.96)

    print(comp_allclose(pytorch_output.last_hidden_state, tt_out_to_torch))
    print(pcc_message)

    if does_pass:
        logger.info("Model Passed!")
    else:
        logger.warning("Model Failed!")

    assert does_pass

def test_WhipserModel_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_model(device=device)
    ttm.device.CloseDevice(device)
