from abc import abstractmethod
import torch

import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import tt_lib
from python_api_testing.models.llama.llama_utils import *
from python_api_testing.models.llama.llama_mlp import TtLlamaMLP
from python_api_testing.models.llama.llama_attention import TtLlamaAttention
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama.llama_embeddings import PytorchEmbeddings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
)

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.llama.llama_model import TtLlamaShared, TtLlamaModel


def run_test_Llama_inference(
    device,
    model_version,
    tokenizer_version,
    batch,
    seq_len,
    num_decoders,
    max_position_embeddings,
    on_weka,
    pcc,
):
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # get only llama model (no linear layer at the end)
    llama_model = hugging_face_reference_model.get_decoder()

    batch = batch
    seq_len = seq_len
    if 1:
        llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)] * batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    pytorch_out = llama_model(llama_input)
    pytorch_out = pytorch_out.last_hidden_state

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    num_decoders = num_decoders
    base_url = "model.layers"
    max_position_embeddings = max_position_embeddings

    tt_llama_model = TtLlamaModel(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders,
    )

    tt_out = tt_llama_model(llama_input).cpu()
    tt_out = tt2torch_tensor(tt_out)
    tt_out = tt_out.squeeze(1)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Llama Model Passed!")
    else:
        logger.warning("Llama Model Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, num_decoders, max_position_embeddings, on_weka, pcc",
    (
        (
            "decapoda-research/llama-7b-hf",
            "hf-internal-testing/llama-tokenizer",
            1,
            128,
            32,
            2048,
            False,
            0.98,
        ),
    ),
)
def test_Llama_inference(
    model_version,
    tokenizer_version,
    batch,
    seq_len,
    num_decoders,
    max_position_embeddings,
    on_weka,
    pcc,
):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    run_test_Llama_inference(
        device,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        num_decoders,
        max_position_embeddings,
        on_weka,
        pcc,
    )
    tt_lib.device.CloseDevice(device)
