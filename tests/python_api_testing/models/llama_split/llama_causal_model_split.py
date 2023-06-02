import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import math
import time
from abc import abstractmethod
import torch
from torch import nn
import tt_lib
from loguru import logger
from python_api_testing.models.llama.llama_utils import (
    tt2torch_tensor,
    torch2tt_tensor,
    linear,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama_split.llama_model_split import (
    TtLlamaModelFirstHalf,
    TtLlamaModelSecondHalf,
    build_decoders,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class TtLlamaCausalModelSecondHalf(TtLlamaModelSecondHalf):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        num_decoders_start,
        num_decoders,
    ):
        # initialize super class
        super().__init__(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            config,
            num_decoders_start,
            num_decoders,
        )

        self.state_dict = state_dict

        # get weights for linear layer
        self.weight = torch2tt_tensor(self.state_dict["lm_head.weight"], self.device)
        self.bias = None

    def forward(self, x, att_mask):
        print("SUper model forward")
        decoder_output = super().forward(x, att_mask)

        print("Aplly linear layer")
        # apply linear layer
        causallm_output = linear(decoder_output, self.weight, self.bias)

        return causallm_output


def run_llama_split_inference(
    state_dict,
    configuration,
    base_url,
    max_position_embeddings,
    num_decoders_start=0,
    num_decoders=2,
    x_input=None,
    half=1,
):
    att_mask = None
    if half == 1:
        tt_llama_model = TtLlamaModelFirstHalf(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(x_input, att_mask)
    else:
        tt_llama_model = TtLlamaCausalModelSecondHalf(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(x_input, att_mask)

    tt_out = tt2torch_tensor(tt_out)
    return tt_out


if __name__ == "__main__":
    torch.manual_seed(1234)

    # parameters -------------------------------------------------------------
    tokenizer_name = "hf-internal-testing/llama-tokenizer"
    llama_model = "decapoda-research/llama-7b-hf"

    base_url = "model.layers"
    max_position_embeddings = 2048
    batch = 1
    seq_len = 32

    first_decoder_start = 0
    second_decoder_start = 16
    num_consecutive_decoders = 16

    # generate input tensor --------------------------------------------------------
    if 1:
        llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)] * batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    # load hugging face llama model ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(llama_model)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config

    state_dict = hugging_face_reference_model.state_dict()

    # execute PyTorch model
    pytorch_out = hugging_face_reference_model(llama_input)
    pytorch_out = pytorch_out.logits
    logger.info(f"Pytorch output shape: {pytorch_out.shape}")

    # Execute TT model -------------------------------------------------------------
    # The first call ----------------------------------------
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()
    first_out = run_llama_split_inference(
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=llama_input,
        half=1,
    )

    tt_lib.device.CloseDevice(device)

    # The second call ----------------------------------------
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = torch2tt_tensor(first_out, device)

    host = tt_lib.device.GetHost()
    tt_out = run_llama_split_inference(
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_input,
        half=2,
    )
    tt_lib.device.CloseDevice(device)
    tt_out = tt_out.squeeze(1)

    logger.info(f"TT output shape: {tt_out.shape}")

    # check outputs --------------------------------------------------------------
    pcc = 0.98
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Llama Causallm output Passed!")
    else:
        logger.warning("Llama Causallm output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"
