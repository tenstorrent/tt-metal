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

from transformers import WhisperModel, WhisperConfig, WhisperPreTrainedModel

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.whisper.whisper_encoder import TtWhisperEncoder, TtWhisperEncoderOutput
from libs import tt_lib as ttm

from python_api_testing.fused_ops.layernorm import Layernorm as TtLayernorm
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_encoder(device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = model.config

    # Changing model configuration
    print("******* Change config *******")
    configuration.max_source_positions = 1024
    configuration.encoder_layers = 4
    print(f"Max source pos {configuration.max_source_positions}, Encodrs {configuration.encoder_layers}")

    # Initializing a model (with random weights) from the changed tiny-style configuration
    model = WhisperModel(configuration)
    pytorch_model = model.encoder
    pytorch_model.eval()
    base_address = "encoder"
    state_dict = model.state_dict()

    batch = 1
    feature_size = 80
    # original from HF example should be: seq_len = 3000, when max_source_positions=1500
    seq_len = 2048

    input_features = torch.rand((batch, feature_size, seq_len))
    head_mask = torch.ones(configuration.encoder_layers, configuration.encoder_attention_heads)

    with torch.no_grad():

        pytorch_output = pytorch_model(
            input_features = input_features,
            head_mask = head_mask,
            output_attentions = True,
            output_hidden_states = True
        )

        print("****** TTM WhisperEncoder ******")

        tt_whisper_encoder = TtWhisperEncoder(
            base_address=base_address,
            state_dict=state_dict,
            device=device,
            config=pytorch_model.config
        )
        tt_whisper_encoder.eval()

        ttm_output = tt_whisper_encoder(
            input_features = input_features,
            head_mask = head_mask,
            output_attentions = True,
            output_hidden_states = True
        )

        ttm_output_to_torch = tt2torch_tensor(ttm_output.last_hidden_state)
        ttm_output_to_torch = torch.squeeze(ttm_output_to_torch, 0)
        does_pass, pcc_message = comp_pcc(pytorch_output.last_hidden_state, ttm_output_to_torch, 0.98)

        print(comp_allclose(pytorch_output.last_hidden_state, ttm_output_to_torch))
        print(pcc_message)

        assert does_pass

        if does_pass:
            logger.info("Encoder Passed!")
        else:
            logger.warning("Encoder Failed!")

def test_whipser_encoder():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_encoder(device=device)
    ttm.device.CloseDevice(device)
