# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.experimental.bert_tiny.tt.bert import TtBert

from transformers import BertForQuestionAnswering, BertTokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.98),),
)
def test_bert_inference(
    pcc,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tokenizer_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    context = [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = ["What discipline did Winkelmann create?"]
    bert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    state_dict = hugging_face_reference_model.state_dict()
    tt_ouptut_model = TtBert(
        hugging_face_reference_model.config, state_dict=state_dict, device=device, mem_config=output_mem_config
    )
    pytorch_output_model = hugging_face_reference_model.bert
    pytorch_out = pytorch_output_model(**bert_input).last_hidden_state

    tt_input = ttnn.to_device(ttnn.from_torch(bert_input.input_ids, dtype=ttnn.uint32), device=device)
    tt_token_ids = ttnn.to_device(ttnn.from_torch(bert_input.token_type_ids, dtype=ttnn.uint32), device=device)
    tt_attention_mask = ttnn.to_device(ttnn.from_torch(bert_input.attention_mask, dtype=ttnn.bfloat16), device=device)

    tt_output = tt_ouptut_model(input_ids=tt_input, token_type_ids=tt_token_ids, attention_mask=tt_attention_mask)
    tt_output = ttnn.to_torch(ttnn.from_device(tt_output)).squeeze(0)

    passing, pcc_message = comp_pcc(pytorch_out, tt_output, pcc)

    logger.info(comp_allclose(pytorch_out, tt_output))
    logger.info(pcc_message)

    if passing:
        logger.info("Bert Passed!")
    else:
        logger.warning("Bert Failed!")

    assert passing, f"Bert does not meet PCC requirement {pcc}."
