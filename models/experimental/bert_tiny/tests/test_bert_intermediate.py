# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.experimental.bert_tiny.tt.bert_intermediate import TtBertintermediate
from transformers import BertForQuestionAnswering
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bert_intermediate_inference(
    pcc,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    state_dict = hugging_face_reference_model.state_dict()
    encoder_idx = 0

    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    tt_intermediate_model = TtBertintermediate(
        hugging_face_reference_model.config,
        encoder_idx,
        state_dict,
        device,
        mem_config=output_mem_config,
    )
    pytorch_intermediate_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].intermediate

    input = (torch.rand(1, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_intermediate_model(input).squeeze(0)

    tt_input = ttnn.to_device(ttnn.from_torch(input.squeeze(1), dtype=ttnn.bfloat16), device=device)
    tt_output = tt_intermediate_model(tt_input)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passing, pcc_message = comp_pcc(pytorch_out, tt_output, pcc)

    logger.info(comp_allclose(pytorch_out, tt_output))
    logger.info(pcc_message)

    if passing:
        logger.info("Bert-tiny_Intermediate Passed!")
    else:
        logger.warning("Bert-tiny_Intermediate Failed!")

    assert passing, f"Bert-tiny_Intermediate output does not meet PCC requirement {pcc}."
