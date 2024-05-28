# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
from loguru import logger
import torch
from torch import nn
import pytest

from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from typing import Optional, Tuple
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import VisionEncoderDecoderModel
from models.experimental.functional_trocr.reference.functional_torch_trocr import (
    TrOCRAttention,
    TrOCRDecoderLayer,
    TrOCRDecoder,
    TrOCRForCausalLM,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_attention(pcc, reset_seeds):
    hidden_states = torch.randn(1, 4, 1024)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    config = model.decoder.config
    model = model.decoder.model.decoder.layers[0].self_attn.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    decoder_l_output = TrOCRAttention(config=config, hidden_states=hidden_states, parameters=parameters)

    model_output = model(hidden_states)
    passing, pcc_message = comp_pcc(model_output[0], decoder_l_output[0], 0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_decoder_layer(pcc, reset_seeds):
    # decoder layer test
    hidden_states = torch.randn(1, 4, 1024)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    config = model.decoder.config
    model = model.decoder.model.decoder.layers[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    decoder_l_output = TrOCRDecoderLayer(config=config, hidden_states=hidden_states, parameters=parameters)[0]

    model_output = model(hidden_states)[0]

    passing, pcc_message = comp_pcc(model_output, decoder_l_output, 0.99)

    logger.info(comp_allclose(model_output, decoder_l_output))
    logger.info(pcc_message)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_decoder(pcc, reset_seeds):
    input_ids = torch.rand((1, 1, 1, 1)).long()
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    config = model.decoder.config
    model = model.decoder.model.decoder
    model_output = model(input_ids)[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    decoder_l_output = TrOCRDecoder(config=config, input_ids=input_ids, parameters=parameters)

    passing, pcc_message = comp_pcc(model_output, decoder_l_output, 0.99)
    logger.info(f"PCC: {pcc_message}")


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_causallm(pcc, reset_seeds):
    input_ids = torch.rand((1, 1, 1, 1)).long()
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    config = model.decoder.config.hidden_size
    model = model.decoder

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    attention_mask = torch.rand((1, 1, 1, 1)).long()
    encoder_hidden_states = torch.rand(1, 577, 768)
    use_cache = False
    output_attentions = False
    output_hidden_states = False
    return_dict = True
    model_output = model(
        input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )[0]
    decoder_l_output = TrOCRForCausalLM(
        config=config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        parameters=parameters,
    )

    passing, pcc_message = comp_pcc(model_output, decoder_l_output, 0.99)
    logger.info(f"PCC: {pcc_message}")
