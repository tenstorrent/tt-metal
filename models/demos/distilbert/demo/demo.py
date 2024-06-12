# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from models.demos.distilbert.tt import ttnn_optimized_distilbert

from models.demos.distilbert.distilbert_utils import (
    squadv2_1K_samples_input,
    squadv2_answer_decode_batch,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)

from ttnn.model_preprocessing import *
from transformers import DistilBertForQuestionAnswering, AutoTokenizer, pipeline

import evaluate


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


def run_distilbert_question_and_answering_inference(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    distilbert,
    model_location_generator,
    input_path,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config
    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    if distilbert == ttnn_optimized_distilbert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {distilbert}")

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: DistilBertForQuestionAnswering.from_pretrained(model_name, torchscript=False).eval(),
        custom_preprocessor=distilbert.custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    context, question = load_inputs(input_path, batch_size)

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters(max_seq_len=sequence_size, padding="max_length")

    inputs = nlp._args_parser({"question": question, "context": context})
    preprocessed_inputs = []
    for i in range(batch_size):
        model_input = next(nlp.preprocess(inputs[0][i], **preprocess_params))
        single_input = {
            "example": model_input["example"],
            "inputs": model_input,
        }
        preprocessed_inputs.append(single_input)

    distilbert_input = tokenizer(
        question,
        context,
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    profiler.start(f"preprocessing_input")
    position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    input_ids, position_ids, attention_mask = distilbert.preprocess_inputs(
        distilbert_input["input_ids"],
        position_ids,
        distilbert_input["attention_mask"],
        device=device,
    )
    profiler.end(f"preprocessing_input")

    mask_reshp = (batch_size, 1, 1, attention_mask.shape[1])
    score_shape = (batch_size, 12, 384, 384)

    mask = (distilbert_input["attention_mask"] == 0).view(mask_reshp).expand(score_shape)
    min_val = torch.zeros(score_shape)
    min_val_tensor = min_val.masked_fill(mask, torch.tensor(torch.finfo(torch.bfloat16).min))

    negative_val = torch.zeros(score_shape)
    negative_val_tensor = negative_val.masked_fill(mask, -1)

    min_val_tensor = ttnn.from_torch(min_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    negative_val_tensor = ttnn.from_torch(negative_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    profiler.start(f"inference_time")
    tt_output = ttnn_optimized_distilbert.distilbert_for_question_answering(
        config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        parameters=parameters,
        device=device,
        min_val_tensor=min_val_tensor,
        negative_val_tensor=negative_val_tensor,
    )
    profiler.end(f"inference_time")

    tt_output = ttnn.to_torch(ttnn.from_device(tt_output)).reshape(batch_size, 1, sequence_size, -1).to(torch.float32)

    tt_start_logits = tt_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_output[..., :, 1].squeeze(1)
    model_answers = {}
    profiler.start("post_processing_output_to_string")
    for i in range(batch_size):
        tt_res = {
            "start": tt_start_logits[i],
            "end": tt_end_logits[i],
            "example": preprocessed_inputs[i]["example"],
            **preprocessed_inputs[i]["inputs"],
        }

        tt_answer = nlp.postprocess([tt_res], **postprocess_params)

        logger.info(f"answer: {tt_answer['answer']}\n")
        model_answers[i] = tt_answer["answer"]

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


def run_distilbert_question_and_answering_inference_squad_v2(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    distilbert,
    model_location_generator,
    n_iterations,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config

    if distilbert == ttnn_optimized_distilbert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {distilbert}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: DistilBertForQuestionAnswering.from_pretrained(model_name, torchscript=False).eval(),
        custom_preprocessor=distilbert.custom_preprocessor,
        device=device,
    )

    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    attention_mask = True
    token_type_ids = False
    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, sequence_size, attention_mask, token_type_ids, batch_size)
    squad_metric = evaluate.load("squad_v2")
    position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    with torch.no_grad():
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            if i < n_iterations:
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]
                ttnn_distilbert_inputs = distilbert.preprocess_inputs(
                    batch_data["input_ids"],
                    position_ids,
                    batch_data["attention_mask"],
                    device=device,
                )
                mask_reshp = (batch_size, 1, 1, batch_data["attention_mask"].shape[1])
                score_shape = (batch_size, 12, 384, 384)

                mask = (batch_data["attention_mask"] == 0).view(mask_reshp).expand(score_shape)
                min_val = torch.zeros(score_shape)
                min_val_tensor = min_val.masked_fill(mask, torch.tensor(torch.finfo(torch.bfloat16).min))

                negative_val = torch.zeros(score_shape)
                negative_val_tensor = negative_val.masked_fill(mask, -1)

                min_val_tensor = ttnn.from_torch(min_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                negative_val_tensor = ttnn.from_torch(negative_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                tt_output = ttnn_optimized_distilbert.distilbert_for_question_answering(
                    config,
                    input_ids=ttnn_distilbert_inputs[0],
                    attention_mask=ttnn_distilbert_inputs[2],
                    position_ids=ttnn_distilbert_inputs[1],
                    parameters=parameters,
                    device=device,
                    min_val_tensor=min_val_tensor,
                    negative_val_tensor=negative_val_tensor,
                )
                tt_output = (
                    ttnn.to_torch(ttnn.from_device(tt_output))
                    .reshape(batch_size, 1, sequence_size, -1)
                    .to(torch.float32)
                )
                cpu_output = hugging_face_reference_model(**batch_data)
                references = batch[1]
                question = batch[2]
                context = batch[3]

                cpu_predictions, tt_predictions = squadv2_answer_decode_batch(
                    hugging_face_reference_model,
                    tokenizer,
                    nlp,
                    references,
                    cpu_output,
                    tt_output,
                    curr_batch_size,
                    question,
                    context,
                )
                pred_labels.extend(tt_predictions)
                cpu_pred_labels.extend(cpu_predictions)
                true_labels.extend(references)

                del tt_output
            i += 1
        eval_score = squad_metric.compute(predictions=pred_labels, references=true_labels)
        cpu_eval_score = squad_metric.compute(predictions=cpu_pred_labels, references=true_labels)
        logger.info(f"\tTT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"\tCPU_Eval: exact: {cpu_eval_score['exact']} -- F1:  {cpu_eval_score['f1']}")


@pytest.mark.parametrize(
    "model_name, input_loc",
    ((["distilbert-base-uncased-distilled-squad", "models/demos/distilbert/demo/input_data.json"]),),
)
@pytest.mark.parametrize("distilbert", [ttnn_optimized_distilbert])
def test_demo(
    input_loc,
    model_name,
    distilbert,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_distilbert_question_and_answering_inference(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        distilbert=distilbert,
        model_location_generator=model_location_generator,
        input_path=input_loc,
    )


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("distilbert", [ttnn_optimized_distilbert])
@pytest.mark.parametrize(
    "n_iterations",
    ((3),),
)
def test_demo_squadv2(
    model_name,
    distilbert,
    n_iterations,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_distilbert_question_and_answering_inference_squad_v2(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        distilbert=distilbert,
        model_location_generator=model_location_generator,
        n_iterations=n_iterations,
    )
