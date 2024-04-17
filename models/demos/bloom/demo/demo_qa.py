# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import torch
import pytest
import evaluate
from loguru import logger
from models.utility_functions import (
    profiler,
    disable_compilation_reports,
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import BloomTokenizerFast, BloomConfig, BloomForQuestionAnswering

from models.demos.bloom.dataset_utils import squadv2_1K_samples_input
from models.demos.bloom.tt.ttnn_optimized_functional_bloom import *
from models.demos.bloom.tt import ttnn_functional_bloom, ttnn_optimized_functional_bloom


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."

        context = []
        question = []
        for i in range(batch):
            context.append(input_data[i]["context"])
            question.append(input_data[i]["question"])

        return context, question


def run_bloom_qa_inference(
    model_version,
    functional_model,
    batch_size,
    input_path,
    model_location_generator,
    device,
    sequence_size,
    reset_seeds,
):
    config = BloomConfig.from_pretrained(model_version)
    tokenizer = BloomTokenizerFast.from_pretrained(model_version)
    hf_ref_model = BloomForQuestionAnswering.from_pretrained(model_version).eval()

    num_heads = config.n_head

    context, question = load_inputs(input_path, batch_size)

    bloom_inputs = tokenizer.batch_encode_plus(
        list(zip(question, context)),
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    profiler.start("preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=f"ttnn_functional_bloom_for_question_answering",
        initialize_model=lambda: hf_ref_model,
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
    )
    profiler.end("preprocessing_parameter")

    input_ids = bloom_inputs["input_ids"]
    attention_mask = bloom_inputs["attention_mask"]

    profiler.start(f"preprocessing_input")
    input_ids, alibi, causal_mask = functional_model.preprocess_inputs(
        input_ids=input_ids, device=device, num_heads=num_heads, attention_mask=attention_mask, max_length=sequence_size
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    tt_output = functional_model.bloom_for_question_answering(
        config, input_ids, alibi, causal_mask, parameters=parameters
    )
    profiler.end(f"inference_time")

    tt_output = ttnn.to_torch(ttnn.from_device(tt_output)).reshape(batch_size, 1, sequence_size, -1).to(torch.float32)

    tt_start_logits = tt_output[..., :, 0].squeeze(0)
    tt_end_logits = tt_output[..., :, 1].squeeze(0)

    profiler.start("post_processing_output_to_string")
    for i in range(batch_size):
        tt_start_index = tt_start_logits[i].argmax(dim=-1).item()
        tt_end_index = tt_end_logits[i].argmax(dim=-1).item()
        answer_tokens = bloom_inputs.input_ids[i, tt_start_index : tt_end_index + 1]
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        logger.info(f"Question: {question[i]}")
        logger.info(f"Answer: {answer_text}")

    profiler.end("post_processing_output_to_string")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }

    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements


def run_bloom_qa_inference_squad(
    model_version,
    functional_model,
    batch_size,
    device,
    sequence_size,
    reset_seeds,
    n_iterations,
):
    config = BloomConfig.from_pretrained(model_version)
    tokenizer = BloomTokenizerFast.from_pretrained(model_version)
    hf_ref_model = BloomForQuestionAnswering.from_pretrained(model_version).eval()

    num_heads = config.n_head

    parameters = preprocess_model_parameters(
        model_name=f"ttnn_functional_bloom_for_question_answering",
        initialize_model=lambda: hf_ref_model,
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
    )

    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, sequence_size, True, False, batch_size)

    with torch.no_grad():
        itr = 0
        predicted_answers = []
        reference_answers = []

        for batch in inputs_squadv2:
            if itr < n_iterations:
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]

                input_ids, alibi, causal_mask = functional_model.preprocess_inputs(
                    input_ids=batch_data["input_ids"],
                    device=device,
                    num_heads=num_heads,
                    attention_mask=batch_data["attention_mask"],
                    max_length=sequence_size,
                )
                tt_output = functional_model.bloom_for_question_answering(
                    config, input_ids, alibi, causal_mask, parameters=parameters
                )

                tt_output = (
                    ttnn.to_torch(ttnn.from_device(tt_output))
                    .reshape(batch_size, 1, sequence_size, -1)
                    .to(torch.float32)
                )

                tt_start_logits = tt_output[..., :, 0].squeeze(0)
                tt_end_logits = tt_output[..., :, 1].squeeze(0)

                reference = batch[1]
                question = batch[2]
                context = batch[3]
                prediction = []

                for idx in range(curr_batch_size):
                    tt_start_index = tt_start_logits[idx].argmax(dim=-1).item()
                    tt_end_index = tt_end_logits[idx].argmax(dim=-1).item()
                    answer_tokens = batch_data.input_ids[idx, tt_start_index : tt_end_index + 1]
                    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

                    prediction.append(
                        {
                            "prediction_text": answer_text,
                            "id": reference[idx]["id"],
                            "no_answer_probability": 0.0,
                        }
                    )
                predicted_answers.extend(prediction)
                reference_answers.extend(reference)

                itr += 1

        squad_metric = evaluate.load("squad_v2")
        eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answers)

        logger.info("Exact Match :")
        logger.info(eval_score["exact"])
        logger.info("F1 Score :")
        logger.info(eval_score["f1"])


@pytest.mark.parametrize(
    "functional_model",
    (
        ttnn_functional_bloom,
        ttnn_optimized_functional_bloom,
    ),
)
def test_demo(
    input_path,
    functional_model,
    model_location_generator,
    device,
    reset_seeds,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bloom_qa_inference(
        model_version="bigscience/bloom-560m",
        functional_model=functional_model,
        batch_size=8,
        input_path=input_path,
        model_location_generator=model_location_generator,
        device=device,
        sequence_size=384,
        reset_seeds=reset_seeds,
    )


@pytest.mark.parametrize(
    "functional_model",
    (
        ttnn_functional_bloom,
        ttnn_optimized_functional_bloom,
    ),
)
@pytest.mark.parametrize(
    "n_iterations",
    ((5),),
)
def test_demo_squadv2(
    functional_model,
    device,
    reset_seeds,
    use_program_cache,
    n_iterations,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bloom_qa_inference_squad(
        model_version="bigscience/bloom-560m",
        functional_model=functional_model,
        batch_size=8,
        device=device,
        sequence_size=384,
        reset_seeds=reset_seeds,
        n_iterations=n_iterations,
    )
