# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
import evaluate
from loguru import logger

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from datasets import load_dataset
from models import generation_utils
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig
from models.demos.grayskull.functional_bloom.tt.ttnn_optimized_functional_bloom import *
from models.demos.grayskull.functional_bloom.tt import ttnn_functional_bloom, ttnn_optimized_functional_bloom


def generate_next_token(
    model, config, input_ids, parameters, num_heads, logits_processor, max_length, attention_mask, **kwargs
):
    num_tokens = input_ids.shape[-1]
    padded_input_ids, alibi, causal_mask = model.preprocess_inputs(
        input_ids=input_ids,
        num_heads=num_heads,
        max_length=max_length,
        attention_mask=attention_mask,
        **kwargs,
    )
    logits = model.bloom_for_causal_lm(
        config=config, input_ids=padded_input_ids, alibi=alibi, causal_mask=causal_mask, parameters=parameters
    )
    next_token_logits = logits[:, num_tokens - 1, :]  # Get the logits for the last token
    processed_logits = logits_processor(input_ids, next_token_logits)
    next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)
    return next_token


def generate(
    model,
    config,
    input_ids,
    parameters,
    tokenizer,
    logits_processor,
    num_heads,
    num_tokens_to_decode,
    attention_mask=None,
    max_length=384,
    **kwargs,
):
    # Tokenize the input text and get initial input_ids
    for _ in range(num_tokens_to_decode):
        next_token = generate_next_token(
            model,
            config,
            input_ids,
            parameters,
            num_heads,
            logits_processor,
            max_length,
            attention_mask,
            **kwargs,
        )

        # Check if the next token is the end-of-sequence token
        if torch.all(next_token == tokenizer.eos_token_id):
            break

        if attention_mask != None:
            attention_mask = torch.cat((attention_mask, torch.ones(next_token.shape, dtype=torch.long)), dim=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids


def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    input_prompt = []
    for i in range(batch):
        context = user_input[i]["context"]
        question = user_input[i]["question"]
        input_prompt.append(f"context:{context} question:{question} Answer:")
    return input_prompt


def run_bloom_qa_inference(
    model_version,
    functional_model,
    batch_size,
    input_path,
    model_location_generator,
    device,
    num_tokens_to_decode,
    reset_seeds,
):
    config = BloomConfig.from_pretrained(model_version)
    tokenizer = BloomTokenizerFast.from_pretrained(model_version)
    hf_ref_model = BloomForCausalLM.from_pretrained(model_version).eval()

    # load input
    input_text = load_inputs(input_path, batch_size)

    num_heads = config.n_head

    profiler.start("preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=f"ttnn_functional_bloom_for_question_answering",
        initialize_model=lambda: hf_ref_model,
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )
    profiler.end("preprocessing_parameter")

    bloom_inputs = tokenizer.batch_encode_plus(
        input_text,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = bloom_inputs["input_ids"]
    attention_mask = bloom_inputs["attention_mask"]

    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_ids = generate(
        model=functional_model,
        config=config,
        input_ids=input_ids,
        parameters=parameters,
        tokenizer=tokenizer,
        logits_processor=logits_processor,
        num_heads=num_heads,
        num_tokens_to_decode=num_tokens_to_decode,
        attention_mask=attention_mask,
        device=device,
    )

    profiler.start("post_processing_output_to_string")
    generated_text = []
    gen_answers = []
    for i in range(len(generated_ids)):
        generated_text.append(tokenizer.decode(generated_ids[i], skip_special_tokens=True))
    profiler.end("post_processing_output_to_string")

    for i in range(len(input_ids)):
        logger.info("Input Prompt")
        logger.info(input_text[i])
        logger.info("Output Prompt")
        input_prompt_length = len(input_text[i])
        answer = generated_text[i][input_prompt_length:].strip()
        gen_answers.append(answer)
        logger.info(answer)

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }

    return measurements, gen_answers


def run_bloom_qa_inference_squad(
    model_version,
    functional_model,
    batch_size,
    device,
    num_tokens_to_decode,
    reset_seeds,
):
    config = BloomConfig.from_pretrained(model_version)
    tokenizer = BloomTokenizerFast.from_pretrained(model_version)
    hf_ref_model = BloomForCausalLM.from_pretrained(model_version).eval()

    num_heads = config.n_head

    squad_dataset = load_dataset("squad_v2")
    validation_split = squad_dataset["validation"]
    predicted_answers = []
    reference_answer = []
    decoded_tt_output = []

    parameters = preprocess_model_parameters(
        model_name=f"ttnn_functional_bloom_for_question_answering",
        initialize_model=lambda: hf_ref_model,
        custom_preprocessor=functional_model.custom_preprocessor,
        device=device,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )

    question = []
    context = []
    answers = []
    id = []

    index = 0
    iter = 0
    while index < batch_size:
        answer = validation_split["answers"][iter]
        if len(answer["text"]) > 0:
            question.append(validation_split["question"][iter])
            context.append(validation_split["context"][iter])
            answers.append(validation_split["answers"][iter])
            id.append(validation_split["id"][iter])
            index += 1
        iter += 1

    input_sentance = [f"context: {c} question: {q} Answer:" for q, c in zip(question, context)]

    bloom_inputs = tokenizer.batch_encode_plus(
        input_sentance,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = bloom_inputs["input_ids"]
    attention_mask = bloom_inputs["attention_mask"]

    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_ids = generate(
        model=functional_model,
        config=config,
        input_ids=input_ids,
        parameters=parameters,
        tokenizer=tokenizer,
        logits_processor=logits_processor,
        num_heads=num_heads,
        num_tokens_to_decode=num_tokens_to_decode,
        attention_mask=attention_mask,
        device=device,
    )

    for i in range(len(generated_ids)):
        decoded_tt_output.append(tokenizer.decode(generated_ids[i], skip_special_tokens=True))

        predicted_answers.append(
            {
                "prediction_text": decoded_tt_output[i],
                "id": id[i],
                "no_answer_probability": 0.0,
            }
        )
        reference_answer.append(
            {
                "answers": {
                    "answer_start": [answers[i]["answer_start"][0]],
                    "text": [answers[i]["text"][0]],
                },
                "id": id[i],
            }
        )

    squad_metric = evaluate.load("squad_v2")
    eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answer)
    logger.info("Exact Match :")
    logger.info(eval_score["exact"])
    logger.info("F1 Score :")
    logger.info(eval_score["f1"])

    return eval_score


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
    use_program_cache,
    reset_seeds,
    batch_size=8,
    num_tokens_to_decode=10,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bloom_qa_inference(
        model_version="bigscience/bloom-560m",
        functional_model=functional_model,
        batch_size=batch_size,
        input_path=input_path,
        model_location_generator=model_location_generator,
        device=device,
        num_tokens_to_decode=num_tokens_to_decode,
        reset_seeds=reset_seeds,
    )


@pytest.mark.parametrize(
    "functional_model",
    (
        ttnn_functional_bloom,
        ttnn_optimized_functional_bloom,
    ),
)
def test_demo_squadv2(
    functional_model,
    device,
    use_program_cache,
    reset_seeds,
    batch_size=8,
    num_tokens_to_decode=10,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bloom_qa_inference_squad(
        model_version="bigscience/bloom-560m",
        functional_model=functional_model,
        batch_size=batch_size,
        device=device,
        num_tokens_to_decode=num_tokens_to_decode,
        reset_seeds=reset_seeds,
    )
