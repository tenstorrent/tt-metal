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
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


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

    def forward(self, x, y, half=1, is_causal=False):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        if half == 2:
            result = self.final_layernorm(result)
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

    def forward(self, x, y, is_causal=False):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        # layer norm is always present in HF pytorch model
        result = self.final_layer_norm(result)

        if is_causal:
            result = self.linear_layer(result)

        return result


def run_llama_split_inference(
    device,
    state_dict,
    configuration,
    base_url,
    max_position_embeddings,
    num_decoders_start,
    num_decoders,
    x_input=None,
    half=1,
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
        tt_out = tt_llama_model(x=x_input, y=position_ids, half=2, is_causal=is_causal)

    tt_out = tt2torch_tensor(tt_out)

    return tt_out


if __name__ == "__main__":
    torch.manual_seed(1234)
    # parameters
    model_version = "decapoda-research/llama-7b-hf"
    tokenizer_version = "hf-internal-testing/llama-tokenizer"
    base_url = "model.layers"
    batch = 1
    seq_len = 128
    max_position_embeddings = 2048
    on_weka = False
    pcc = 0.98
    is_causal = False

    first_decoder_start = 0
    second_decoder_start = 16
    num_consecutive_decoders = 16

    decoder_stack_list = [i for i in range(2 * num_consecutive_decoders + 1)]

    # Prepare input =============================================================
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1

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
    logger.info(f"PyTorch output shape: {pytorch_out.shape}")

    # TT hardware execution ============================================================
    # TT execution ======================================================================
    # The first call --------------------------
    first_call_start = time.time()
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # prepare input for TT hardware
    tt_llama_input = llama_input.unsqueeze(1)
    tt_llama_input = torch2tt_tensor(tt_llama_input, device)

    first_out = run_llama_split_inference(
        device,
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_llama_input,
        half=1,
    )

    tt_lib.device.CloseDevice(device)
    first_call_end = time.time()
    logger.info(f"First call duration: {first_call_end - first_call_start}")

    # The second call -------------------------------------------------------
    second_call_start = time.time()
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = torch2tt_tensor(first_out, device)

    tt_out = run_llama_split_inference(
        device,
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_input,
        half=2,
        is_causal=is_causal,
    )

    tt_lib.device.CloseDevice(device)
    second_call_end = time.time()
    logger.info(f"Second call duration: {second_call_end - second_call_start}")

    tt_out = tt_out.squeeze(1)

    logger.info(f"PY out shape: {pytorch_out.shape}")
    logger.info(f"TT out shape: {tt_out.shape}")

    # check outputs ---------------------------------------------------------
    pcc = 0.98
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Stacked Decoders test Passed!")
    else:
        logger.warning("Stacked Decoders test Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"
