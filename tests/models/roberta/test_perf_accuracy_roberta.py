# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForQuestionAnswering
import evaluate
import pytest
from models.utility_functions import torch_to_tt_tensor_rm
from tests.models.roberta.roberta_for_question_answering import (
    TtRobertaForQuestionAnswering,
)

from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
)
from models.utility_functions import prep_report

BATCH_SIZE = 1


def run_demo_roberta(model_name, expected_inference_time, expected_compile_time, iterations, device):
    squad_dataset = load_dataset("squad_v2")

    disable_persistent_kernel_cache()

    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)
    model.eval()

    tt_model = TtRobertaForQuestionAnswering(
        config=model.config,
        base_address=f"",
        device=device,
        state_dict=model.state_dict(),
        reference_model=model,
    )
    tt_model.eval()

    with torch.no_grad():
        question, context = (
            "What discipline did Winkelmann create?",
            "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art.",
        )
        profiler.start(cpu_key)
        input = tokenizer(question, context, return_tensors="pt")
        generated_output = model(**input)
        torch_answer_start_index = generated_output.start_logits.argmax()
        torch_answer_end_index = generated_output.end_logits.argmax()

        torch_predict_answer_tokens = input.input_ids[0, torch_answer_start_index : torch_answer_end_index + 1]
        torch_answer = tokenizer.decode(torch_predict_answer_tokens, skip_special_tokens=True)
        profiler.end(cpu_key)

        tt_attention_mask = torch.unsqueeze(input.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch_to_tt_tensor_rm(tt_attention_mask, device)

        profiler.start(first_key)
        tt_output = tt_model(input.input_ids, tt_attention_mask)

        tt_answer_start_index = tt_output.start_logits.argmax()
        tt_answer_end_index = tt_output.end_logits.argmax()

        tt_predict_answer_tokens = input.input_ids[0, tt_answer_start_index : tt_answer_end_index + 1]
        tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_answer

        enable_persistent_kernel_cache()

        profiler.start(second_key)

        tt_output = tt_model(input.input_ids, tt_attention_mask)

        tt_answer_start_index = tt_output.start_logits.argmax()
        tt_answer_end_index = tt_output.end_logits.argmax()

        tt_predict_answer_tokens = input.input_ids[0, tt_answer_start_index : tt_answer_end_index + 1]
        tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_answer

    validation_split = squad_dataset["validation"]
    predicted_answers = []
    reference_answer = []
    index = 0
    iteration = 0

    profiler.start(third_key)
    while iteration < iterations:
        question = validation_split["question"][index]
        context = validation_split["context"][index]
        answers = validation_split["answers"][index]
        id = validation_split["id"][index]
        index += 1
        if len(answers["text"]) > 0:
            iteration += 1
            input = tokenizer(question, context, return_tensors="pt")

            tt_attention_mask = torch.unsqueeze(input.attention_mask, 0)
            tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
            tt_attention_mask = torch_to_tt_tensor_rm(tt_attention_mask, device, put_on_device=False)

            tt_output = tt_model(input.input_ids, tt_attention_mask)

            tt_answer_start_index = tt_output.start_logits.argmax()
            tt_answer_end_index = tt_output.end_logits.argmax()

            tt_predict_answer_tokens = input.input_ids[0, tt_answer_start_index : tt_answer_end_index + 1]
            tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)

            predicted_answers.append(
                {
                    "prediction_text": tt_answer,
                    "id": id,
                    "no_answer_probability": 0.0,
                }
            )
            reference_answer.append(
                {
                    "answers": {
                        "answer_start": [answers["answer_start"][0]],
                        "text": [answers["text"][0]],
                    },
                    "id": id,
                }
            )
            del tt_answer
    tt_lib.device.Synchronize()
    profiler.end(third_key)

    squad_metric = evaluate.load("squad_v2")
    eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answer)

    logger.info("Exact Match :")
    logger.info(eval_score["exact"])
    logger.info("F1 Score :")
    logger.info(eval_score["f1"])

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)

    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name=f"roberta-base-squad2",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"Roberta",
        inference_time_cpu=cpu_time,
    )

    logger.info(f"{model_name} inference time: {second_iter_time}")
    logger.info(f"{model_name} compile time: {compile_time}")
    logger.info(f"{model_name} inference for {iteration} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration",
    (
        (
            "deepset/roberta-base-squad2",
            0.98,
            17.5,
            50,
        ),
    ),
)
def test_demo_bare_metal_roberta(
    model_name, use_program_cache, expected_inference_time, expected_compile_time, iteration, device
):
    run_demo_roberta(
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration",
    (
        (
            "deepset/roberta-base-squad2",
            1.4,
            17.8,
            50,
        ),
    ),
)
def test_demo_virtual_machine_roberta(
    model_name, use_program_cache, expected_inference_time, expected_compile_time, iteration, device
):
    run_demo_roberta(
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        device,
    )
