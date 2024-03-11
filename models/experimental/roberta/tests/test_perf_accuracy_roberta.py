# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import torch
import pytest
import evaluate

from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForQuestionAnswering

from models.utility_functions import (
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.roberta.roberta_common import torch2tt_tensor
from models.experimental.roberta.tt.roberta_for_question_answering import TtRobertaForQuestionAnswering

BATCH_SIZE = 1


def run_perf_roberta(expected_inference_time, expected_compile_time, device, iterations):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "Question-Answering"
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"

    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    hf_model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    hf_model.eval()

    tt_model = TtRobertaForQuestionAnswering(
        config=hf_model.config,
        state_dict=hf_model.state_dict(),
        base_address="",
        device=device,
        reference_model=hf_model,
    )
    tt_model.eval()

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    input = tokenizer(question, context, return_tensors="pt")

    tt_attn_mask = torch.unsqueeze(input.attention_mask, 0)
    tt_attn_mask = torch.unsqueeze(tt_attn_mask, 0)
    tt_attn_mask = torch2tt_tensor(tt_attn_mask, device)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = hf_model(**input)

        torch_answer_start_index = torch_output.start_logits.argmax()
        torch_answer_end_index = torch_output.end_logits.argmax()

        torch_predict_answer_tokens = input.input_ids[0, torch_answer_start_index : torch_answer_end_index + 1]
        torch_answer = tokenizer.decode(torch_predict_answer_tokens, skip_special_tokens=True)

        tt_lib.device.Synchronize(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(input.input_ids, tt_attn_mask)

        tt_answer_start_index = tt_output.start_logits.argmax()
        tt_answer_end_index = tt_output.end_logits.argmax()

        tt_predict_answer_tokens = input.input_ids[0, tt_answer_start_index : tt_answer_end_index + 1]
        tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)

        tt_lib.device.Synchronize(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(input.input_ids, tt_attn_mask)

        tt_answer_start_index = tt_output.start_logits.argmax()
        tt_answer_end_index = tt_output.end_logits.argmax()

        tt_predict_answer_tokens = input.input_ids[0, tt_answer_start_index : tt_answer_end_index + 1]
        tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)

        tt_lib.device.Synchronize(device)
        profiler.end(second_key)
        del tt_output

        squad_dataset = load_dataset("squad_v2")
        validation_split = squad_dataset["validation"]
        predicted_answers = []
        reference_answers = []
        iteration = 0
        index = 0

        profiler.start(third_key)
        while iteration < iterations:
            question = validation_split["question"][index]
            context = validation_split["context"][index]
            answers = validation_split["answers"][index]
            id = validation_split["id"][index]

            inputs = tokenizer(question, context, return_tensors="pt")

            tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
            tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
            tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

            index += 1

            if len(answers["text"]) > 0:
                iteration += 1
                tt_output = tt_model(inputs.input_ids, tt_attention_mask)

                tt_answer_start_index = tt_output.start_logits.argmax()
                tt_answer_end_index = tt_output.end_logits.argmax()

                tt_predict_answer_tokens = inputs.input_ids[0, tt_answer_start_index : tt_answer_end_index + 1]
                tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)

                prediction_answer = {"id": id, "prediction_text": tt_answer, "no_answer_probability": 0.0}
                predicted_answers.append(prediction_answer)

                reference_answer = {
                    "id": id,
                    "answers": {"text": [answers["text"][0]], "answer_start": [tt_answer_start_index]},
                }
                reference_answers.append(reference_answer)

        squad_metric = evaluate.load("squad_v2")
        eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answers)

        logger.info("Exact Match :")
        logger.info(eval_score["exact"])
        logger.info("F1 Score :")
        logger.info(eval_score["f1"])

        tt_lib.device.Synchronize(device)
        profiler.end(third_key)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)

    prep_perf_report(
        model_name="roberta",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time

    logger.info(f"roberta {comments} inference time: {second_iter_time}")
    logger.info(f"roberta compile time: {compile_time}")
    logger.info(f"roberta inference time for {iteration} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            0.405,
            17,
            100,
        ),
    ),
)
def test_perf_bare_metal(device, use_program_cache, expected_inference_time, expected_compile_time, iterations):
    run_perf_roberta(expected_inference_time, expected_compile_time, device, iterations)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            0.60,
            17.5,
            100,
        ),
    ),
)
def test_perf_virtual_machine(device, use_program_cache, expected_inference_time, expected_compile_time, iterations):
    run_perf_roberta(expected_inference_time, expected_compile_time, device, iterations)
