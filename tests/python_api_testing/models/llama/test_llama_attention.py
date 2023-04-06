import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
from loguru import logger
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from libs import tt_lib as ttl
from typing import List, Optional, Tuple, Union

from transformers import T5Tokenizer, T5Model, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm
from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TTsoftmax
from python_api_testing.models.llama.llama_utils import *
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.llama.llama_attention import TtLlamaAttention


class PytorchLlamaAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.model.layers[layer_num].self_attn

        # Disable dropout
        self.attention.eval()

    def forward(self, x, y):
        result = self.attention(hidden_states=x, position_ids=y)[0]
        return result


def run_test_LlamaAttention_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):

    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare inputs ========================================================================
    torch.manual_seed(0)
    # hidden states tensor: batch size is equal to 32, sequence length: 32
    attention_input = (torch.rand(batch, seq_len, 4096) * 2) - 1
    layer_num = 0
    base_url = 'model.layers'
    # max_position_embeddings parameter should be in the config file, but the used pretrained model doesn't consist this parameter
    max_position_embeddings = 2048

    # get positions_ids values
    seq_length = attention_input.shape[1]
    past_key_values_length = 0

    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=None
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # PyTorch output =======================================================================
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaAttention_model(x=attention_input, y=position_ids)

    # TT hardware execution =================================================================
    tt_attention_input = attention_input.unsqueeze(1)
    tt_attention_input = torch2tt_tensor(tt_attention_input, device)

    # get TT Attention module
    tt_LlamaAttention_model = TtLlamaAttention(
        device,
        base_url,
        state_dict,
        layer_num,
        configuration.hidden_size,
        configuration.num_attention_heads,
        max_position_embeddings
    )

    tt_out, attn_weights, past_key_value = tt_LlamaAttention_model(hidden_states=tt_attention_input, position_ids=position_ids)
    tt_out = tt2torch_tensor(tt_out).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    print(comp_allclose(pytorch_out, tt_out))
    print(comp_pcc(pytorch_out, tt_out))

    passing_pcc, output_pcc = comp_pcc(pytorch_out, tt_out, 0.98)

    assert passing_pcc, "PCC value is lower than 0.98"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, on_weka, pcc",
    (
        ("decapoda-research/llama-7b-hf", "hf-internal-testing/llama-tokenizer", 32, 128, False, 0.98),
    ),
)
def test_LlamaAttention_inference(model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    # host = ttl.device.GetHost()
    run_test_LlamaAttention_inference(
        device,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        on_weka,
        pcc
    )
    ttl.device.CloseDevice(device)
