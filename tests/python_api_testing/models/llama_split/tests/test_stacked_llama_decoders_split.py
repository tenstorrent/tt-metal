import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import time
from loguru import logger
import torch
import numpy as np
from torch import nn, Tensor
import tt_lib
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

from python_api_testing.models.llama.llama_utils import *
from python_api_testing.models.llama.llama_mlp import TtLlamaMLP
from python_api_testing.models.llama.llama_attention import TtLlamaAttention
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from utility_functions_new import comp_pcc
from python_api_testing.models.llama_split.llama_split_utils import gen_position_ids


class TtLlamaDecoderModelStacked(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        start,
        count,
    ):
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings
        self.config = config

        self.decoder_list = torch.nn.Sequential(
            *[
                TtLlamaDecoderLayer(
                    self.device,
                    self.state_dict,
                    self.base_url,
                    decoder_idx,
                    self.max_position_embeddings,
                    self.config,
                )
                for decoder_idx in range(start, start + count)
            ]
        )

        # add final normalization layer
        self.layer_num = None
        self.layer_position = "norm"
        self.final_layernorm = TtLlamaRMSNorm(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=self.layer_num,
            layer_position=self.layer_position,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # if it is CausalLM Llama model
        self.weight = torch2tt_tensor(self.state_dict[f"lm_head.weight"], self.device)
        self.bias = None

    def forward(self, x, y, half=1, has_layer_norm=False, is_causal=False):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        if half == 2:
            # add norm layer
            if has_layer_norm:
                result = self.final_layernorm(result)
            # add linear
            if is_causal:
                result = linear(result, self.weight, self.bias)

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
        # get final norm layer
        self.final_layer_norm = hf_reference_model.model.norm
        # Disable dropout
        self.final_layer_norm.eval()

        # get linear layer
        self.linear_layer = hf_reference_model.lm_head
        self.linear_layer.eval()

    def forward(self, x, y, has_layer_norm=False, is_causal=False):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        # layer norm is always present in HF pytorch model
        if has_layer_norm:
            result = self.final_layer_norm(result)

        if is_causal:
            result = self.linear_layer(result)

        return result


def run_test_llama_decoder_split_inference(
    device,
    state_dict,
    configuration,
    base_url,
    max_position_embeddings,
    num_decoders_start,
    num_decoders,
    x_input=None,
    position_ids=None,
    half=1,
    has_layer_norm=False,
    is_causal=False,
):
    if half == 1:
        logger.info(f"First half of the model execution")
        tt_llama_model = TtLlamaDecoderModelStacked(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(x=x_input, y=position_ids)
    else:
        logger.info(f"Second half of the model execution")
        tt_llama_model = TtLlamaDecoderModelStacked(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(
            x=x_input,
            y=position_ids,
            half=2,
            has_layer_norm=has_layer_norm,
            is_causal=is_causal,
        )

    tt_out = tt2torch_tensor(tt_out)

    return tt_out


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_batch = 1
_seq_len = 32
_max_position_embeddings = 2048
# is causallm llama model (it has additional linear layer at the end)
_on_weka = False

# how many decoders to use
# number of decoders to be stacked started from the selected id in the original llama model
# e.g. stack 16 consecutive decoders
_num_consecutive_decoders = 4

# decoder id from which decoder stacking starts (the first half of the model)
# e.g. start from 0 add use 3 decoders (0, 1, and 2)
_first_decoder_start = 0

# decoder id from which decoder stacking starts (the second half of the model)
# e.g. start from 16 add use 3 decoders (16, 17, and 18)
_second_decoder_start = _num_consecutive_decoders

# has_layer_norm - add norm layer after stacked decoders
# is_causal - add linear layer after decoders
# parameters --------------------------------------------------


@pytest.mark.parametrize(
    "pcc, has_layer_norm, is_causal",
    (
        (
            0.98,
            False,
            False,
        ),
    ),
)
def test_llama_decoder_split_inference(
    pcc,
    has_layer_norm,
    is_causal,
):
    # set parameters ================================================================
    model_version = _llama_model_name
    tokenizer_version = _tokenizer_name
    base_url = _base_url
    batch = _batch
    seq_len = _seq_len
    max_position_embeddings = _max_position_embeddings
    on_weka = _on_weka
    first_decoder_start = _first_decoder_start
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    # Set number
    decoder_stack_list = [i for i in range(2 * num_consecutive_decoders + 1)]

    # Prepare input =============================================================
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1

    # get positions_ids values
    position_ids = gen_position_ids(llama_input)

    # Load hugging face model ========================================================
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # PyTorch output ==================================================================
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModelStacked(
        hugging_face_reference_model, decoder_stack_list
    )
    pytorch_LlamaDecoder_model.eval()

    # get output
    pytorch_out = pytorch_LlamaDecoder_model(
        x=llama_input, y=position_ids, is_causal=is_causal
    )

    # TT hardware execution ============================================================
    # The first call --------------------------
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # prepare input for TT hardware
    tt_llama_input = llama_input.unsqueeze(1)
    tt_llama_input = torch2tt_tensor(tt_llama_input, device)

    first_out = run_test_llama_decoder_split_inference(
        device,
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_llama_input,
        position_ids=position_ids,
        half=1,
    )

    tt_lib.device.CloseDevice(device)

    # The second call -------------------------------------------------------
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = torch2tt_tensor(first_out, device)

    tt_out = run_test_llama_decoder_split_inference(
        device,
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_input,
        position_ids=position_ids,
        half=2,
        has_layer_norm=has_layer_norm,
        is_causal=is_causal,
    )

    tt_lib.device.CloseDevice(device)

    # squeeze output
    tt_out = tt_out.squeeze(1)

    logger.debug(f"Pytorch output shape: {pytorch_out.shape}")
    logger.debug(f"Tenstorrent output shape: {tt_out.shape}")

    # check outputs ---------------------------------------------------------
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Stacked Decoders test Passed!")
    else:
        logger.warning("Stacked Decoders test Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"
