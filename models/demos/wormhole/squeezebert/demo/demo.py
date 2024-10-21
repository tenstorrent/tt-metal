# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import json
import torch
import pytest
import evaluate
import transformers
from loguru import logger
from models.utility_functions import (
    profiler,
    is_wormhole_b0,
    skip_for_grayskull,
    disable_compilation_reports,
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.squeezebert.tt import ttnn_functional_squeezebert
from models.datasets.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch


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


def run_squeezebert_question_and_answering_inference(
    mesh_device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    squeezebert,
    input_path,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = transformers.SqueezeBertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )

    state_dict = hugging_face_reference_model.state_dict()
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = weights_mesh_mapper = output_mesh_composer = None
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    if mesh_device_flag:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        batch_size = 16
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            profiler.start(f"preprocessing_parameter")
            parameters = preprocess_model_parameters(
                model_name=tt_model_name,
                initialize_model=lambda: hugging_face_reference_model,
                custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
                device=mesh_device,
            )
            profiler.end(f"preprocessing_parameter")

    else:
        mesh_device = ttnn.open_device(device_id=0) if is_wormhole_b0() else mesh_device
        profiler.start(f"preprocessing_parameter")
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: hugging_face_reference_model,
            custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
            device=mesh_device,
        )
        profiler.end(f"preprocessing_parameter")

    tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config
    nlp = transformers.pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

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

    squeezebert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    profiler.start(f"preprocessing_input")
    position_ids = positional_ids(config, squeezebert_input.input_ids)
    ttnn_squeezebert_inputs = squeezebert.preprocess_inputs(
        squeezebert_input["input_ids"],
        squeezebert_input["token_type_ids"],
        position_ids,
        squeezebert_input["attention_mask"],
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    tt_output = squeezebert.squeezebert_for_question_answering(
        config,
        *ttnn_squeezebert_inputs,
        state_dict=state_dict,
        base_addr=f"transformer.",
        parameters=parameters,
        device=mesh_device,
        reader_patterns_cache={},
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )
    profiler.end(f"inference_time")

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer).to(torch.float32)

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


def run_squeezebert_question_and_answering_inference_squad_v2(
    mesh_device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    squeezebert,
    n_iterations,
):
    disable_persistent_kernel_cache()
    hugging_face_reference_model = transformers.SqueezeBertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )

    state_dict = hugging_face_reference_model.state_dict()
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = weights_mesh_mapper = output_mesh_composer = None
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    if mesh_device_flag:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        batch_size = 16
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            parameters = preprocess_model_parameters(
                model_name=tt_model_name,
                initialize_model=lambda: hugging_face_reference_model,
                custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
                device=mesh_device,
            )

    else:
        mesh_device = ttnn.open_device(device_id=0) if is_wormhole_b0() else mesh_device
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: hugging_face_reference_model,
            custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
            device=mesh_device,
        )

    tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config

    nlp = transformers.pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

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

                ttnn_squeezebert_inputs = squeezebert.preprocess_inputs(
                    batch_data["input_ids"],
                    batch_data["token_type_ids"],
                    position_ids,
                    batch_data["attention_mask"],
                    device=mesh_device,
                    mesh_mapper=inputs_mesh_mapper,
                )

                tt_output = squeezebert.squeezebert_for_question_answering(
                    config,
                    *ttnn_squeezebert_inputs,
                    state_dict=state_dict,
                    base_addr=f"transformer.",
                    parameters=parameters,
                    device=mesh_device,
                    reader_patterns_cache={},
                    mesh_mapper=inputs_mesh_mapper,
                    mesh_composer=output_mesh_composer,
                )
                tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer).to(torch.float32)

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
        # logger.info(f"\tCPU_Eval: exact: {cpu_eval_score['exact']} --  F1: {cpu_eval_score['f1']}")


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_name, input_loc",
    ((["squeezebert/squeezebert-uncased", "models/demos/wormhole/squeezebert/demo/input_data.json"]),),
)
@pytest.mark.parametrize("squeezebert", [ttnn_functional_squeezebert])
def test_demo(input_loc, model_name, squeezebert, mesh_device, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_squeezebert_question_and_answering_inference(
        mesh_device=mesh_device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        squeezebert=squeezebert,
        input_path=input_loc,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("squeezebert", [ttnn_functional_squeezebert])
@pytest.mark.parametrize(
    "n_iterations",
    ((3),),
)
def test_demo_squadv2(model_name, squeezebert, n_iterations, mesh_device, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_squeezebert_question_and_answering_inference_squad_v2(
        mesh_device=mesh_device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        squeezebert=squeezebert,
        n_iterations=n_iterations,
    )
