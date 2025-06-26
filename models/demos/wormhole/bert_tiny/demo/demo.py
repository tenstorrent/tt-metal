# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json

import evaluate
import pytest
import torch
from loguru import logger
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.datasets.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch
from models.demos.wormhole.bert_tiny.tt.bert_tiny import bert_for_question_answering, preprocess_inputs
from models.utility_functions import disable_persistent_kernel_cache, is_wormhole_b0, profiler, skip_for_grayskull


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


def positional_ids(config, input_ids, past_key_values_length=0):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0)[:, past_key_values_length : seq_length + past_key_values_length]
    position_ids = position_ids.expand_as(input_ids)

    return position_ids


def run_bert_question_and_answering_inference(
    mesh_device,
    model_name,
    sequence_size,
    model_location_generator,
    input_path,
):
    disable_persistent_kernel_cache()
    model = str(model_location_generator(model_name, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model, torchscript=False)
    pytorch_model = hugging_face_reference_model.eval()

    tokenizer_name = str(model_location_generator(model_name, model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    config = hugging_face_reference_model.config
    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    profiler.start(f"preprocessing_parameter")
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    profiler.end(f"preprocessing_parameter")

    context, question = load_inputs(input_path, batch_size)

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
    preprocess_params["max_seq_len"] = sequence_size
    inputs = nlp._args_parser({"context": context, "question": question})
    preprocessed_inputs = []
    for i in range(batch_size):
        model_input = next(nlp.preprocess(inputs[0][i], **preprocess_params))
        single_input = {
            "example": model_input["example"],
            "inputs": model_input,
        }
        preprocessed_inputs.append(single_input)

    bert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    position_ids = positional_ids(config, bert_input.input_ids)
    profiler.start(f"preprocessing_input")
    ttnn_bert_inputs = preprocess_inputs(
        bert_input["input_ids"],
        bert_input["token_type_ids"],
        position_ids,
        bert_input["attention_mask"],
        mesh_device=mesh_device,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    ttnn_output = bert_for_question_answering(
        config,
        *ttnn_bert_inputs,
        parameters=parameters,
        device=mesh_device,
    )
    profiler.end(f"inference_time")

    ttnn_output = (
        ttnn.to_torch(ttnn.from_device(ttnn_output), mesh_composer=output_mesh_composer)
        .reshape(batch_size, 1, sequence_size, -1)
        .to(torch.float32)
    )

    ttnn_start_logits = ttnn_output[..., :, 0].squeeze(1)
    ttnn_end_logits = ttnn_output[..., :, 1].squeeze(1)

    model_answers = {}
    profiler.start("post_processing_output_to_string")
    for i in range(batch_size):
        tt_res = {
            "start": ttnn_start_logits[i],
            "end": ttnn_end_logits[i],
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


def run_bert_question_and_answering_inference_squad_v2(
    mesh_device,
    model_name,
    sequence_size,
    model_location_generator,
    n_iterations,
):
    disable_persistent_kernel_cache()

    model = str(model_location_generator(model_name, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model, torchscript=False)
    pytorch_model = hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer_name = str(model_location_generator(model_name, model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    config = hugging_face_reference_model.config

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    attention_mask = True
    token_type_ids = True
    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, sequence_size, attention_mask, token_type_ids, batch_size)
    squad_metric = evaluate.load("squad_v2")

    with torch.no_grad():
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            if i < n_iterations:
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]
                position_ids = positional_ids(config, batch_data.input_ids)
                ttnn_bert_inputs = preprocess_inputs(
                    batch_data["input_ids"],
                    batch_data["token_type_ids"],
                    position_ids,
                    batch_data["attention_mask"],
                    mesh_device=mesh_device,
                    inputs_mesh_mapper=inputs_mesh_mapper,
                )
                tt_output = bert_for_question_answering(
                    config,
                    *ttnn_bert_inputs,
                    parameters=parameters,
                    device=mesh_device,
                )
                tt_output = (
                    ttnn.to_torch(ttnn.from_device(tt_output), mesh_composer=output_mesh_composer)
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
        logger.info(f"TT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"CPU_Eval: exact: {cpu_eval_score['exact']} -- F1:  {cpu_eval_score['f1']}")


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("model_name", ["mrm8488/bert-tiny-finetuned-squadv2"])
@pytest.mark.parametrize("input_loc", ["models/demos/wormhole/bert_tiny/demo/input_data.json"])
def test_demo(
    input_loc,
    sequence_size,
    model_name,
    model_location_generator,
    mesh_device,
):
    disable_persistent_kernel_cache()

    return run_bert_question_and_answering_inference(
        mesh_device=mesh_device,
        model_name=model_name,
        sequence_size=sequence_size,
        model_location_generator=model_location_generator,
        input_path=input_loc,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("model_name", ["mrm8488/bert-tiny-finetuned-squadv2"])
@pytest.mark.parametrize(
    "n_iterations",
    ((1),),
)
def test_demo_squadv2(
    model_name,
    sequence_size,
    n_iterations,
    model_location_generator,
    mesh_device,
):
    disable_persistent_kernel_cache()

    return run_bert_question_and_answering_inference_squad_v2(
        mesh_device=mesh_device,
        model_name=model_name,
        sequence_size=sequence_size,
        model_location_generator=model_location_generator,
        n_iterations=n_iterations,
    )
