# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.experimental.bert_tiny.tt.bert_encoder import TtBertencoder

from transformers import BertForQuestionAnswering
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bert_encoder_inference(
    pcc,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    state_dict = hugging_face_reference_model.state_dict()
    tt_ouptut_model = TtBertencoder(
        hugging_face_reference_model.config, state_dict=state_dict, device=device, mem_config=output_mem_config
    )
    pytorch_output_model = hugging_face_reference_model.bert.encoder
    input = (torch.rand(1, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_output_model(input.squeeze(1))[0]

    tt_input = ttnn.to_device(ttnn.from_torch(input, dtype=ttnn.bfloat16), device=device)
    tt_output = tt_ouptut_model(tt_input)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(pytorch_out, tt_output, pcc)

    logger.info(comp_allclose(pytorch_out, tt_output))
    logger.info(pcc_message)

    if passing:
        logger.info("Bert_Encoder Passed!")
    else:
        logger.warning("Bert_Encoder Failed!")

    assert passing, f"Bert_Encoder output does not meet PCC requirement {pcc}."
