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

from transformers import WhisperModel, WhisperConfig

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.whisper.whisper_decoder import TtWhisperDecoder, TtWhisperDecoderOutput

from libs import tt_lib as ttm

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

def run_whisper_decoder(device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    base_address = "decoder"

    configuration = model.config
    pytorch_model = model.decoder
    pytorch_model.eval()
    state_dict = model.state_dict()

    batch_size = 1
    seq_len = 32

    # (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    # Indices of input sequence tokens in the vocabulary
    decoder_input_ids = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1]]
    ) * configuration.decoder_start_token_id

    pytorch_output = pytorch_model(
        input_ids = decoder_input_ids,
        output_attentions = True,
        output_hidden_states = True
    )

    print("****** TTM WhisperDecoder ******")

    tt_whisper_decoder = TtWhisperDecoder(
        base_address = base_address,
        state_dict = state_dict,
        device = device,
        config = model.config
    )
    tt_whisper_decoder.eval()

    ttm_output = tt_whisper_decoder(
        input_ids = decoder_input_ids,
        output_attentions = True,
        output_hidden_states = True
    )

    # Check last_hidden_state
    ttm_output_to_torch = tt2torch_tensor(ttm_output.last_hidden_state)
    ttm_output_to_torch = torch.squeeze(ttm_output_to_torch, 0)

    does_pass, pcc_message = comp_pcc(pytorch_output.last_hidden_state, ttm_output_to_torch, 0.98)

    print(comp_allclose(pytorch_output.last_hidden_state, ttm_output_to_torch))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Decoder Passed!")
    else:
        logger.warning("Decoder Failed!")

def test_whipser_decoder():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_decoder(device=device)
    ttm.device.CloseDevice(device)
