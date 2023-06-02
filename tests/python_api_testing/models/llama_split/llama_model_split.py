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
from python_api_testing.models.llama.llama_utils import tt2torch_tensor, torch2tt_tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def build_decoders(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    config,
    num_decoder_start,
    num_decoders,
):
    decoder_list = torch.nn.Sequential(
        *[
            TtLlamaDecoderLayer(
                device,
                state_dict,
                base_url,
                decoder_idx,
                max_position_embeddings,
                config,
            )
            for decoder_idx in range(
                num_decoder_start, num_decoder_start + num_decoders
            )
        ]
    )
    return decoder_list


class TtLlamaModelFirstHalf(torch.nn.Module):
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
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings
        self.num_decoders_start = num_decoders_start
        self.num_decoders = num_decoders
        self.config = config

        # firt part =================================================================
        self.embeddings = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.embeddings.weight = torch.nn.Parameter(
            state_dict[f"model.embed_tokens.weight"]
        )

        # stack all decoders
        self.decoders_first = build_decoders(
            self.device,
            self.state_dict,
            self.base_url,
            self.max_position_embeddings,
            self.config,
            self.num_decoders_start,
            self.num_decoders,
        )

    def forward(self, x, att_mask):
        print("embedding")
        embeddings = self.embeddings(x)
        print("embedding to tt")
        tt_embeddings = torch2tt_tensor(embeddings, self.device)

        # apply decoders
        print("apply decoders")
        first_encoder_output = tt_embeddings
        for idx, decoder_layer in enumerate(self.decoders_first):
            print(f"deocder call: {idx}")
            first_encoder_output = decoder_layer(
                hidden_states=first_encoder_output, attention_mask=att_mask
            )[0]

        print("return")
        # return only hidden_states
        return first_encoder_output


class TtLlamaModelSecondHalf(torch.nn.Module):
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
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings
        self.num_decoders_start = num_decoders_start
        self.num_decoders = num_decoders
        self.config = config

        # stack all decoders
        self.decoders_second = build_decoders(
            self.device,
            self.state_dict,
            self.base_url,
            self.max_position_embeddings,
            self.config,
            self.num_decoders_start,
            self.num_decoders,
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
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, x, att_mask):
        second_encoder_output = x
        for idx, decoder_layer in enumerate(self.decoders_second):
            print(f"decoder in second: {idx}")
            second_encoder_output = decoder_layer(
                hidden_states=second_encoder_output, attention_mask=att_mask
            )[0]

        print("Apply LayerNorm")
        # apply final norm layer
        norm_encoder_output = self.final_layernorm(second_encoder_output)

        return norm_encoder_output


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
        first_model_create_start = time.time()
        tt_llama_model = TtLlamaModelFirstHalf(
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
        tt_out = tt_llama_model(x_input, att_mask)
        end = time.time()
        logger.info(f"First half - inference duration: {end-start}")
        # FIRST create model: 446.87756752967834
        # FIRST INFERERENCE: 35.20861005783081
    else:
        second_model_create_start = time.time()
        tt_llama_model = TtLlamaModelSecondHalf(
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
        tt_out = tt_llama_model(x_input, att_mask)
        end = time.time()
        logger.info(f"Second half - inference duration: {end-start}")
        # SECOND create model: 449.55821108818054
        # SECOND INFERENCE: 25.62920570373535

    tt_out = tt2torch_tensor(tt_out)
    return tt_out


if __name__ == "__main__":
    torch.manual_seed(1234)

    # parameters
    base_url = "model.layers"
    max_position_embeddings = 2048
    batch = 1
    seq_len = 32

    first_decoder_start = 0
    second_decoder_start = 16
    num_consecutive_decoders = 16

    # generate input tensor ---------------------------------------------------
    if 1:
        llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)] * batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    # load hugging face llama model -------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf"
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config

    state_dict = hugging_face_reference_model.state_dict()
    # get only decoder part for llama model
    hugging_face_reference_model = hugging_face_reference_model.get_decoder()

    # execute PyTorch model
    pytorch_out = hugging_face_reference_model(llama_input)
    pytorch_out = pytorch_out.last_hidden_state
    logger.info(f"Pytorch output shape: {pytorch_out.shape}")

    # Execute TT model --------------------------------------------------------
    # The first call ---------------------
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

    # The second call ---------------------
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = torch2tt_tensor(first_out, device)

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
