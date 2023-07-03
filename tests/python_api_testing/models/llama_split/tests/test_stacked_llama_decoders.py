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
from utility_functions_new import comp_pcc

from python_api_testing.models.llama_split.tt.stacked_decoders import (
    TtLlamaDecoderModelStacked,
)
from python_api_testing.models.llama_split.reference.cpu_stacked_decoders import (
    PytorchLlamaDecoderModelStacked,
)


def run_test_llama_decoder_inference(
    device,
    llama_input,
    model_version,
    tokenizer_version,
    base_url,
    batch,
    seq_len,
    max_position_embeddings,
    num_decoders,
    on_weka,
    pcc,
):
    # stack decoders
    start = 0
    decoder_stack_list = [i for i in range(num_decoders + 1)]

    # get positions_ids values
    position_ids = gen_position_ids(llama_input)

    # Load Pytorch model ===================================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        model_version, torch_dtype=torch.float32
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
        start,
        num_decoders,
    )

    tt_out = tt_LlamaDecoder_model(x=tt_llama_input, y=position_ids)

    # transform to PyTorch tensor
    tt_out = tt2torch_tensor(tt_out)
    tt_out = tt_out.squeeze(1)

    # check outputs =========================================================================
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Test for stacked decoders passed!")
    else:
        logger.warning("Test for stacked decoders failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_batch = 1
_seq_len = 32
_max_position_embeddings = 2048
_on_weka = False
_num_decoders = 4
# num_decoders - number of consecutive decoders
# parameters --------------------------------------------------


@pytest.mark.parametrize(
    "pcc",
    ((0.98),),
)
def test_llama_decoder_inference(
    pcc,
):
    # set parameters ================================================================
    model_version = _llama_model_name
    tokenizer_version = _tokenizer_name
    base_url = _base_url
    batch = _batch
    seq_len = _seq_len
    max_position_embeddings = _max_position_embeddings
    on_weka = _on_weka
    num_decoders = _num_decoders

    # Prepare input ========================================================================
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_test_llama_decoder_inference(
        device,
        llama_input,
        model_version,
        tokenizer_version,
        base_url,
        batch,
        seq_len,
        max_position_embeddings,
        num_decoders,
        on_weka,
        pcc,
    )
    tt_lib.device.CloseDevice(device)
