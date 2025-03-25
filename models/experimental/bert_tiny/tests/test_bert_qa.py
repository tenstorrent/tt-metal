# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.experimental.bert_tiny.tt.bert_for_question_answering import TtBertforqa

from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_bert_qa_inference(
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
    state_dict = hugging_face_reference_model.state_dict()
    tt_ouptut_model = TtBertforqa(
        config=hugging_face_reference_model.config,
        state_dict=state_dict,
        device=device,
    )
    pytorch_output_model = hugging_face_reference_model

    question = ["What discipline did Winkelmann create?"]
    bert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    nlp = pipeline(
        "question-answering",
        model=hugging_face_reference_model,
        tokenizer=tokenizer,
    )

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
    preprocess_params["max_seq_len"] = 128
    input_q = {"context": context, "question": question}
    examples = nlp._args_parser(input_q)

    single_inputs = []

    model_input = next(nlp.preprocess(examples[0][0], **preprocess_params))

    tt_input = ttnn.to_device(ttnn.from_torch(model_input["input_ids"], dtype=ttnn.uint32), device=device)
    tt_token_ids = ttnn.to_device(ttnn.from_torch(model_input["token_type_ids"], dtype=ttnn.uint32), device=device)
    tt_attention_mask = ttnn.to_device(
        ttnn.from_torch(model_input["attention_mask"], dtype=ttnn.bfloat16), device=device
    )

    single_input = {
        "data": (
            model_input["input_ids"],
            model_input["attention_mask"],
            model_input["token_type_ids"],
        ),
        "example": model_input["example"],
        "inputs": model_input,
    }
    single_inputs.append(single_input)

    pytorch_out = pytorch_output_model(**bert_input)
    tt_output = tt_ouptut_model(input_ids=tt_input, token_type_ids=tt_token_ids, attention_mask=tt_attention_mask)
    tt_output = ttnn.to_torch(ttnn.from_device(tt_output))

    tt_start_logits = tt_output[..., :, 0].squeeze(1).float()
    tt_end_logits = tt_output[..., :, 1].squeeze(1).float()

    pt_start_logits = pytorch_out.start_logits.detach()
    pt_end_logits = pytorch_out.end_logits.detach()

    passing_start, output = comp_pcc(pt_start_logits, tt_start_logits, pcc)
    logger.info(f"Start Logits {output}")
    _, output = comp_allclose(pt_start_logits, tt_start_logits)
    logger.info(f"Start Logits {output}")

    if not passing_start:
        logger.error(f"Start Logits PCC < {pcc}")

    passing_end, output = comp_pcc(pt_end_logits, tt_end_logits, pcc)
    logger.info(f"End Logits {output}")
    _, output = comp_allclose(
        pt_end_logits,
        tt_end_logits,
    )
    logger.info(f"End Logits {output}")
    if not passing_end:
        logger.error(f"End Logits PCC < {pcc}")

    tt_res = {
        "start": tt_start_logits[0],
        "end": tt_end_logits[0],
        "example": single_inputs[0]["example"],
        **single_inputs[0]["inputs"],
    }

    tt_answer = nlp.postprocess([tt_res], **postprocess_params)
    logger.info(f"TT: {tt_answer}")
    pt_res = {
        "start": pt_start_logits[0],
        "end": pt_end_logits[0],
        "example": single_inputs[0]["example"],
        **single_inputs[0]["inputs"],
    }

    pt_answer = nlp.postprocess([pt_res], **postprocess_params)
    logger.info(f"PT: {pt_answer}")
