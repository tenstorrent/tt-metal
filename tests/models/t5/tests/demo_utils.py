# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
import evaluate

from models.generation_utils import run_generate

from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
)
from models.utility_functions import prep_report

BATCH_SIZE = 1

def run_perf_t5(t5_model_constructor, model_name, expected_inference_time, expected_compile_time, iterations, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    squad_dataset = load_dataset("squad_v2")

    disable_persistent_kernel_cache()

    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    context = "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    question = "What discipline did Winkelmann create?"
    input_text = f"question: {question} context: {context}"

    with torch.no_grad():
        profiler.start(cpu_key)
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512)
        generated_output = model.generate(input_ids)
        generated_answer = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        profiler.end(cpu_key)

        profiler.start(first_key)
        output_sentence = run_generate(
            input_text,
            tokenizer,
            t5_model_constructor,
            device,
            run_tt_model=True,
            log=False,
        )
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del output_sentence

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        output_sentence = run_generate(
            input_text,
            tokenizer,
            t5_model_constructor,
            device,
            run_tt_model=True,
            log=False,
        )
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del output_sentence

    profiler.start(third_key)
    validation_split = squad_dataset["validation"]
    predicted_answers = []
    reference_answer = []
    index = 0
    iteration = 0

    while iteration < iterations:
        question = validation_split["question"][index]
        context = validation_split["context"][index]
        answers = validation_split["answers"][index]
        id = validation_split["id"][index]
        index += 1
        if len(answers["text"]) > 0:
            iteration += 1
            logger.info(iteration)
            input_text = f"question: {question} context: {context}"

            output_sentence = run_generate(
                input_text,
                tokenizer,
                t5_model_constructor,
                device,
                run_tt_model=True,
                log=False,
            )
            predicted_answers.append(
                {
                    "prediction_text": output_sentence,
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
            del output_sentence

    squad_metric = evaluate.load("squad_v2")
    eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answer)

    logger.info("Exact Match :")
    logger.info(eval_score["exact"])
    logger.info("F1 Score :")
    logger.info(eval_score["f1"])
    tt_lib.device.Synchronize()
    profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)

    t5_type = {"t5-base": "t5-base", "google/flan-t5-small": "t5-flan-small"}[model_name]

    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name=t5_type,
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=t5_type,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"{model_name} inference time: {second_iter_time}")
    logger.info(f"{model_name} compile time: {compile_time}")
    logger.info(f"{model_name} inference for {iteration} Samples: {third_iter_time}")
