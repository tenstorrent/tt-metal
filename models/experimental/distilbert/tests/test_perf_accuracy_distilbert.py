# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import evaluate

from loguru import logger
from datasets import load_dataset

from models.utility_functions import (
    Profiler,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from models.experimental.distilbert.tt.distilbert import distilbert_for_question_answering


BATCH_SIZE = 1


def run_perf_distilbert(expected_inference_time, expected_compile_time, device, iterations):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    comments = "question-answering"
    cpu_key = "ref_key"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")

    tt_attn_mask = torch_to_tt_tensor_rm(inputs.attention_mask, device, put_on_device=False)

    tt_model = distilbert_for_question_answering(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(inputs.input_ids, tt_attn_mask)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(inputs.input_ids, tt_attn_mask)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

        profiler.start(third_key)
        squad_dataset = load_dataset("squad_v2")
        validation_split = squad_dataset["validation"]
        predicted_answers = []
        reference_answer = []
        iteration = 0
        index = 0

        while iteration < iterations:
            question = validation_split["question"][index]
            context = validation_split["context"][index]
            answers = validation_split["answers"][index]
            inputs = tokenizer(question, context, return_tensors="pt")
            tt_attn_mask = torch_to_tt_tensor_rm(inputs.attention_mask, device, put_on_device=False)
            id = validation_split["id"][index]
            index += 1

            if len(answers["text"]) > 0:
                iteration += 1
                tt_output = tt_model(inputs.input_ids, tt_attn_mask)

                tt_start_logits_torch = tt_to_torch_tensor(tt_output.start_logits).squeeze(0).squeeze(0)
                tt_end_logits_torch = tt_to_torch_tensor(tt_output.end_logits).squeeze(0).squeeze(0)

                answer_start_index = tt_start_logits_torch.argmax()
                answer_end_index = tt_end_logits_torch.argmax()

                predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
                answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

                prediction_data = {"id": id, "prediction_text": answer, "no_answer_probability": 0.0}
                predicted_answers.append(prediction_data)

                reference_data = {
                    "id": id,
                    "answers": {"text": [answers["text"][0]], "answer_start": [answer_start_index]},
                }
                reference_answer.append(reference_data)

        squad_metric = evaluate.load("squad_v2")
        eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answer)
        logger.info("Exact Match :")
        logger.info(eval_score["exact"])
        logger.info("F1 Score :")
        logger.info(eval_score["f1"])

        ttnn.synchronize_device(device)
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_perf_report(
        model_name="distilbert",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_inference_time,
        expected_inference_time=expected_compile_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"distilbert inference time: {second_iter_time}")
    logger.info(f"distilbert compile time: {compile_time}")
    logger.info(f"distilbert inference for {iterations} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            0.27,
            13.024,
            50,
        ),
    ),
)
def test_perf_bare_metal(device, use_program_cache, expected_inference_time, expected_compile_time, iterations):
    run_perf_distilbert(expected_inference_time, expected_compile_time, device, iterations)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            0.27,
            13.024,
            50,
        ),
    ),
)
def test_perf_virtual_machine(device, use_program_cache, expected_inference_time, expected_compile_time, iterations):
    run_perf_distilbert(expected_inference_time, expected_compile_time, device, iterations)
