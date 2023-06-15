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
import torch
from torch import nn
import tt_lib
from loguru import logger
from python_api_testing.models.llama.llama_utils import tt2torch_tensor, torch2tt_tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama_split.hf_llama_classes import (
    TtLlamaModelFirstHFModel,
    TtLlamaModelSecondHFModel,
)

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from transformers.generation.configuration_utils import GenerationConfig


def run_llama_split_inference(
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
    x_inputs=None,
    position_ids=None,
    half=1,
):
    if half == 1:
        logger.info("First pass throught TT model")
        first_model_create_start = time.time()
        tt_llama_model = TtLlamaModelFirstHFModel(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        first_model_create_end = time.time()
        logger.info(
            f"First half - duration of model creation: {first_model_create_end-first_model_create_start}"
        )
        start = time.time()
        tt_out = tt_llama_model(input_ids=x_inputs, position_ids=position_ids)
        end = time.time()
        logger.info(f"First half - inference duration: {end-start}")
    else:
        logger.info("Second pass throught TT model")
        second_model_create_start = time.time()
        tt_llama_model = TtLlamaModelSecondHFModel(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        second_model_create_end = time.time()
        logger.info(
            f"Second half - duration of model creation: {second_model_create_end-second_model_create_start}"
        )

        start = time.time()
        tt_out = tt_llama_model(input_ids=x_inputs, position_ids=position_ids)
        end = time.time()
        logger.info(f"Second half - inference duration: {end-start}")

    # returned type from the model is tuple
    tt_output = tt2torch_tensor(tt_out[0])
    return tt_output


@pytest.mark.parametrize(
    "model_version, tokenizer_version, base_url, batch, seq_len, max_position_embeddings, first_decoder_start, second_decoder_start, num_consecutive_decoders, on_weka, pcc",
    (
        (
            "decapoda-research/llama-7b-hf",
            "hf-internal-testing/llama-tokenizer",
            "model.layers",
            1,
            32,
            2048,
            0,
            16,
            16,
            False,
            0.98,
        ),
    ),
)
def test_llama_split_inference(
    model_version,
    tokenizer_version,
    batch,
    seq_len,
    max_position_embeddings,
    first_decoder_start,
    second_decoder_start,
    num_consecutive_decoders,
    on_weka,
    pcc,
):
    tt_lib.device.EnableCompileCache()

    torch.manual_seed(1234)
    # first_decoder_start = 0
    # second_decoder_start = 16
    # num_consecutive_decoders = 16

    # parameters
    # base_url = "model.layers"
    # max_position_embeddings = 2048
    # batch = 1
    # seq_len = 32
    # tokenizer_name = "huggyllama/llama-7b"
    # llama_model_name = "huggyllama/llama-7b"
    # is_causallm = False

    # generate input tensor ----------------------------------------------------------
    llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    position_ids = gen_position_ids(llama_input)

    # create llama pytorch model =====================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_version)

    hugging_face_reference_model.eval()

    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # execute PyTorch model
    llama_model = hugging_face_reference_model.get_decoder()
    pytorch_out = llama_model(llama_input)
    pytorch_out = pytorch_out.last_hidden_state
    logger.info(f"Pytorch output shape: {pytorch_out.shape}")

    # Execute TT model ===============================================================

    logger.info(f"The first half started")
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    first_out = run_llama_split_inference(
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=llama_input,
        position_ids=position_ids,
        half=1,
    )
    tt_lib.device.CloseDevice(device)
    logger.info(f"The first half ended")

    # The second call -------------------------------------------------------
    logger.info(f"The second half started")
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    # tt_input = torch2tt_tensor(first_out, device)
    tt_input = first_out

    tt_out = run_llama_split_inference(
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=tt_input,
        position_ids=position_ids,
        half=2,
    )
    logger.info(f"The second half ended")

    # squeeze
    tt_out = tt_out.squeeze(1)

    # check outputs -----------------------------------------------------------
    pcc = 0.98
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Llama Model Passed!")
    else:
        logger.warning("Llama Model Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


# if __name__ == "__main__":
#     torch.manual_seed(1234)
#     first_decoder_start = 0
#     second_decoder_start = 16
#     num_consecutive_decoders = 16

#     # parameters
#     base_url = "model.layers"
#     max_position_embeddings = 2048
#     batch = 1
#     seq_len = 32
#     tokenizer_name = "huggyllama/llama-7b"
#     llama_model_name = "huggyllama/llama-7b"

#     # generate input tensor ----------------------------------------------------------
#     llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)

#     # get positions_ids values
#     past_key_values_length = 0
#     seq_length = llama_input.shape[1]

#     position_ids = torch.arange(
#         past_key_values_length,
#         seq_length + past_key_values_length,
#         dtype=torch.long,
#         device=None,
#     )
#     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

#     # create llama pytorch model =====================================================
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
#         llama_model_name
#     )

#     hugging_face_reference_model.eval()
#     # get configurations
#     configuration = hugging_face_reference_model.config
#     state_dict = hugging_face_reference_model.state_dict()

#     # execute PyTorch model
#     pytorch_out = hugging_face_reference_model(llama_input)
#     pytorch_out = pytorch_out.logits
#     logger.info(f"Pytorch output shape: {pytorch_out.shape}")

#     # Execute TT model ===============================================================

#     logger.info(f"The first half started")
#     device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
#     tt_lib.device.InitializeDevice(device)
#     tt_lib.device.SetDefaultDevice(device)
#     host = tt_lib.device.GetHost()

#     first_out = run_llama_split_inference(
#         state_dict,
#         base_url,
#         max_position_embeddings,
#         configuration,
#         num_decoders_start=first_decoder_start,
#         num_decoders=num_consecutive_decoders,
#         x_inputs=llama_input,
#         position_ids=position_ids,
#         half=1,
#     )
#     tt_lib.device.CloseDevice(device)
#     logger.info(f"The first half ended")

#     # The second call -------------------------------------------------------
#     logger.info(f"The second half started")
#     device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
#     tt_lib.device.InitializeDevice(device)
#     tt_lib.device.SetDefaultDevice(device)

#     # send input tensor from host to tt device
#     # tt_input = torch2tt_tensor(first_out, device)
#     tt_input = first_out

#     tt_out = run_llama_split_inference(
#         state_dict,
#         base_url,
#         max_position_embeddings,
#         configuration,
#         num_decoders_start=second_decoder_start,
#         num_decoders=num_consecutive_decoders,
#         x_inputs=tt_input,
#         position_ids=position_ids,
#         half=2,
#     )
#     logger.info(f"The second half ended")

#     # squeeze
#     tt_out = tt_out.squeeze(1)

#     # check outputs -----------------------------------------------------------
#     pcc = 0.98
#     logger.info(comp_allclose(pytorch_out, tt_out))

#     does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
#     logger.info(f"PCC value: {pcc_value}")

#     if does_pass:
#         logger.info("Llama Model Passed!")
#     else:
#         logger.warning("Llama Model Failed!")
#         assert does_pass, f"PCC value is lower than {pcc}"
