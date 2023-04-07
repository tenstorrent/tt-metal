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
from python_api_testing.models.whisper.whisper_encoder_layer import TtWhisperEncoderLayer

from libs import tt_lib as ttm

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

def run_whisper_encoder_layer(layer, device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    state_dict = model.state_dict()

    configuration = model.config
    embed_dim = configuration.d_model # 384
    encoder_ffn_dim = configuration.encoder_ffn_dim # 1536
    num_heads = configuration.encoder_attention_heads # 6

    """ Huggingface Whisper Encoder Layer """
    ENCODER_IND = layer
    pytorch_model = model.encoder.layers[ENCODER_IND]
    pytorch_model.eval()
    base_address = f"encoder.layers.{ENCODER_IND}"

    batch = 32
    tgt_len = 32
    src_len = 32

    hidden_state_input_tensor = torch.randn(src_len, batch, embed_dim)
    attention_mask_input_tensor = torch.randn(batch, 1, tgt_len, src_len)
    layer_head_mask_input_tensor = torch.randn(num_heads, )

    pytorch_output = pytorch_model(
        hidden_state_input_tensor,
        attention_mask_input_tensor,
        layer_head_mask_input_tensor
    )

    """ TTM Whisper Encoder Layer """

    ttm_tensor_hidden_state = torch2tt_tensor(hidden_state_input_tensor, device)
    ttm_tensor_attention_mask = torch2tt_tensor(attention_mask_input_tensor, device)

    # layer_head_mask_input_tensor has size [6] and is equal to number of encoder_attention_heads
    # Stays torch tensor and then is used in attention mechanism approprietly for now
    # Because can't convert 1d tensor of size [6] to ttm

    tt_whisper_encoder_layer = TtWhisperEncoderLayer(
        base_address = base_address,
        state_dict = state_dict,
        device = device,
        embed_dim = embed_dim,
        num_heads = num_heads,
        encoder_ffn_dim = encoder_ffn_dim
    )
    ttm_output = tt_whisper_encoder_layer(
        ttm_tensor_hidden_state,
        ttm_tensor_attention_mask,
        layer_head_mask_input_tensor,
        True
    )

    # Returns Tuple of size 2
    # Attention weights Not used
    # Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
    # but it is not used

    ttm_output_to_torch_0 = tt2torch_tensor(ttm_output[0])
    ttm_output_to_torch_0 = torch.squeeze(ttm_output_to_torch_0, 0)
    does_pass, pcc_message = comp_pcc(pytorch_output[0], ttm_output_to_torch_0, 0.98)

    print(comp_allclose(pytorch_output[0], ttm_output_to_torch_0))
    print(pcc_message)

    if does_pass:
        logger.info("Encoder layer Passed!")
    else:
        logger.warning("Encoder layer Failed!")

    assert does_pass


def test_WhipserEncoderLayer_inference():
    torch.manual_seed(1234)
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_whisper_encoder_layer(layer=0, device=device)
    ttm.device.CloseDevice(device)
