# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from loguru import logger
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

import ttnn

from models.demos.metal_BERT_large_11.tt.bert_model import TtBertBatchDram
from models.demos.metal_BERT_large_11.tt.model_config import get_model_config, get_tt_cache_path

from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    is_e75,
)


def run_bert_question_and_answering_inference(
    model_version,
    batch,
    seq_len,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
    PERF_CNT,
    device,
):
    torch.manual_seed(1234)

    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    hugging_face_reference_model.eval()
    tt_bert_model = TtBertBatchDram(
        hugging_face_reference_model.config,
        hugging_face_reference_model,
        device,
        model_config,
        tt_cache_path,
    )

    if real_input:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        context = batch * [
            "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
        ]
        question = batch * ["What discipline did Winkelmann create?"]
        bert_input = tokenizer.batch_encode_plus(
            zip(question, context),
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=attention_mask,
            return_token_type_ids=token_type_ids,
            return_tensors="pt",
        )
        nlp = pipeline(
            "question-answering",
            model=hugging_face_reference_model,
            tokenizer=tokenizer,
        )
        pl_answer = nlp(question=question, context=context)

        preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
        preprocess_params["max_seq_len"] = seq_len
        input_q = {"context": context, "question": question}
        examples = nlp._args_parser(input_q)

        single_inputs = []
        for i in range(batch):
            model_input = next(nlp.preprocess(examples[0][i], **preprocess_params))
            single_input = {
                "data": (
                    model_input["input_ids"],
                    model_input["attention_mask"] if attention_mask else None,
                    model_input["token_type_ids"] if token_type_ids else None,
                ),
                "example": model_input["example"],
                "inputs": model_input,
            }
            single_inputs.append(single_input)
    else:
        if 1:
            bert_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
        else:
            # batch identical sequences for debugging
            oneseq = [torch.arange(seq_len)] * batch
            bert_input = torch.stack(oneseq)
            bert_input = bert_input.reshape(batch, seq_len)

    tt_attention_mask = tt_bert_model.model_attention_mask(**bert_input)

    tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**bert_input)

    pytorch_out = hugging_face_reference_model(**bert_input)

    tt_attention_mask = tt_attention_mask.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
    tt_embedding_inputs = {
        key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"]) for (key, value) in tt_embedding_inputs.items()
    }

    tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)
    tt_out = tt_bert_model(tt_embedding, tt_attention_mask).cpu()

    tt_untilized_output = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch().reshape(batch, 1, seq_len, -1).to(torch.float32)

    tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)

    pt_start_logits = pytorch_out.start_logits.detach()
    pt_end_logits = pytorch_out.end_logits.detach()

    passing_start, output = comp_pcc(pt_start_logits, tt_start_logits, pcc)
    logger.info(f"Start Logits {output}")
    _, output = comp_allclose(
        pt_start_logits, tt_start_logits, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Start Logits {output}")
    if not passing_start:
        logger.error(f"Start Logits PCC < {pcc}")

    passing_end, output = comp_pcc(pt_end_logits, tt_end_logits, pcc)
    logger.info(f"End Logits {output}")
    _, output = comp_allclose(
        pt_end_logits, tt_end_logits, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"End Logits {output}")
    if not passing_end:
        logger.error(f"End Logits PCC < {pcc}")

    passing = passing_start and passing_end

    del tt_out

    if model_config["DEFAULT_DTYPE"] == ttnn.bfloat8_b and not passing:
        pytest.xfail("PCC is garbage for BFLOAT8_B. Numbers are for perf only!")

    assert passing, f"At least one start or end logits don't meet PCC requirement {pcc}"


@pytest.mark.parametrize(
    "batch, model_config_str",
    (
        (9, "BFLOAT8_B-DRAM"),
        (9, "BFLOAT16-DRAM"),
        (9, "BFLOAT8_B-L1"),
        (9, "BFLOAT16-L1"),
        (9, "MIXED_PRECISION_BATCH9"),
        (8, "MIXED_PRECISION_BATCH8"),
        (8, "BFLOAT8_B-SHARDED"),
        (7, "BFLOAT8_B-SHARDED"),
        (12, "BFLOAT8_B-SHARDED"),
    ),
    ids=[
        "batch_9-BFLOAT8_B-DRAM",
        "batch_9-BFLOAT16-DRAM",
        "batch_9-BFLOAT8_B-L1",
        "batch_9-BFLOAT16-L1",
        "batch_9-MIXED_PRECISION_BATCH9",
        "batch_8-MIXED_PRECISION_BATCH8",
        "batch_8-BFLOAT8_B-SHARDED",
        "batch_7-BFLOAT8_B-SHARDED",
        "batch_12-BFLOAT8_B-SHARDED",
    ],
)
@pytest.mark.parametrize(
    "model_version, seq_len, real_input, attention_mask, token_type_ids, pcc",
    (
        (
            "phiyodr/bert-large-finetuned-squad2",
            384,
            True,
            True,
            True,
            0.97,
        ),
    ),
    ids=["BERT_LARGE"],
)
def test_bert(
    device,
    use_program_cache,
    model_version,
    batch,
    seq_len,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_config_str,
    model_location_generator,
):
    if is_e75(device):
        pytest.skip(f"Bert large 11 is not supported on E75")

    model_config = get_model_config(batch, device.compute_with_storage_grid_size(), model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    # This test will run BERT-Large once with cache disabled.
    # Then it will enable cache and run BERT-Large PERF_CNT number of times.
    # Performance is reported only for PERF_CNT number of runs.
    PERF_CNT = 1

    run_bert_question_and_answering_inference(
        model_version,
        batch,
        seq_len,
        real_input,
        attention_mask,
        token_type_ids,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
        PERF_CNT,
        device,
    )
