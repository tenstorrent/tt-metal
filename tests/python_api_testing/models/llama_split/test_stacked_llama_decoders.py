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
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import tt_lib
from typing import List, Optional, Tuple, Union

from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from python_api_testing.models.llama.llama_utils import *
from python_api_testing.models.llama.llama_mlp import TtLlamaMLP
from python_api_testing.models.llama.llama_attention import TtLlamaAttention
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class TtLlamaDecoderModelStacked(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        decoder_ids,
    ):
        super().__init__()
        self.decoder_list = torch.nn.Sequential(
            *[
                TtLlamaDecoderLayer(
                    device,
                    state_dict,
                    base_url,
                    decoder_idx,
                    max_position_embeddings,
                    config,
                )
                for decoder_idx in decoder_ids
            ]
        )

    def forward(self, x, y):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        return result


class PytorchLlamaDecoderModelStacked(torch.nn.Module):
    def __init__(self, hf_reference_model, decoder_ids):
        super().__init__()
        self.decoder_list = torch.nn.Sequential(
            *[
                hf_reference_model.model.layers[decoder_idx]
                for decoder_idx in decoder_ids
            ]
        )

    def forward(self, x, y):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        return result


def run_test_LlamaDecoder_inference(
    device, model_version, tokenizer_version, batch, seq_len, num_decoders, on_weka, pcc
):
    # Prepare input ========================================================================
    torch.manual_seed(0)
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1
    base_url = "model.layers"
    # max_position_embeddings parameter should be in the config file,
    # but the used pretrained model doesn't consist this parameter
    max_position_embeddings = 2048
    decoder_stack_list = [i for i in range(num_decoders + 1)]

    # get positions_ids values
    past_key_values_length = 0
    seq_length = llama_input.shape[1]

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=None,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Load Pytorch model ===================================================================
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # PyTorch output =========================================================================
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModelStacked(
        hugging_face_reference_model, decoder_stack_list
    )
    pytorch_LlamaDecoder_model.eval()
    pytorch_out = pytorch_LlamaDecoder_model(x=llama_input, y=position_ids)

    # TT hardware execution =================================================================
    tt_llama_input = llama_input.unsqueeze(1)
    tt_llama_input = torch2tt_tensor(tt_llama_input, device)

    tt_LlamaDecoder_model = TtLlamaDecoderModelStacked(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        decoder_stack_list,
    )

    tt_out = tt_LlamaDecoder_model(x=tt_llama_input, y=position_ids)

    # transform to PyTorch tensor
    tt_out = tt2torch_tensor(tt_out)
    tt_out = tt_out.squeeze(1)

    # check outputs =========================================================================
    pcc = 0.98

    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Test for stacked decoders passed!")
    else:
        logger.warning("Test for stacked decoders failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


if __name__ == "__main__":
    # input parameters
    model_version = "decapoda-research/llama-7b-hf"
    tokenizer_version = "hf-internal-testing/llama-tokenizer"
    batch = 1
    seq_len = 128
    num_decoders = 4
    on_weka = False
    pcc = 0.98

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_test_LlamaDecoder_inference(
        device,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        num_decoders,
        on_weka,
        pcc,
    )
    tt_lib.device.CloseDevice(device)
