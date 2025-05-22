# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json

import evaluate
import pytest
import torch
import transformers
from loguru import logger
from transformers import RobertaForQuestionAnswering, RobertaTokenizer, pipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.datasets.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch
from models.demos.bert.tt import ttnn_optimized_bert
from models.utility_functions import disable_persistent_kernel_cache, profiler


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


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def run_roberta_question_and_answering_inference(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bert,
    input_path,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = RobertaForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config
    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)
    config.use_dram = True

    tt_model_name = f"ttnn_{model_name}_optimized"

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
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

    roberta_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    profiler.start(f"preprocessing_input")

    position_ids = create_position_ids_from_input_ids(
        input_ids=roberta_input.input_ids, padding_idx=config.pad_token_id
    )
    ttnn_roberta_inputs = bert.preprocess_inputs(
        roberta_input["input_ids"],
        roberta_input["token_type_ids"],
        position_ids,
        roberta_input["attention_mask"],
        device=device,
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    tt_output = bert.bert_for_question_answering(
        config,
        *ttnn_roberta_inputs,
        parameters=parameters,
        name="roberta",
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

        logger.info(f"Question: {question[i]}")
        logger.info(f"Answer: {tt_answer['answer']}\n")
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


def run_roberta_question_and_answering_inference_squad_v2(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bert,
    n_iterations,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = RobertaForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config
    config.use_dram = True

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
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
                position_ids = create_position_ids_from_input_ids(
                    input_ids=batch_data.input_ids, padding_idx=config.pad_token_id
                )
                ttnn_roberta_inputs = bert.preprocess_inputs(
                    batch_data["input_ids"],
                    batch_data["token_type_ids"],
                    position_ids,
                    batch_data["attention_mask"],
                    device=device,
                )

                tt_output = bert.bert_for_question_answering(
                    config,
                    *ttnn_roberta_inputs,
                    parameters=parameters,
                    name="roberta",
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
        logger.info(f"TT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"CPU_Eval: exact: {cpu_eval_score['exact']} --  F1: {cpu_eval_score['f1']}")

        assert eval_score["exact"] >= cpu_eval_score["exact"] and eval_score["f1"] >= cpu_eval_score["f1"], (
            f"Expected Exact Match: {cpu_eval_score['exact']}, Actual Exact Match: {eval_score['exact']}; "
            f"Expected F1 Score: {cpu_eval_score['f1']}, Actual F1 Score: {eval_score['f1']}"
        )


@pytest.mark.parametrize(
    "model_name, input_loc",
    ((["deepset/roberta-large-squad2", "models/demos/roberta/demo/input_data.json"]),),
)
@pytest.mark.parametrize(
    ("bert", "batch_size", "sequence_size"),
    ((ttnn_optimized_bert, 8, 384),),
)
def test_demo(device, use_program_cache, model_name, input_loc, bert, batch_size, sequence_size):
    disable_persistent_kernel_cache()

    return run_roberta_question_and_answering_inference(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=batch_size,
        sequence_size=sequence_size,
        bert=bert,
        input_path=input_loc,
    )


@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize(
    ("bert", "batch_size", "sequence_size", "n_iterations"),
    ((ttnn_optimized_bert, 8, 384, 3),),
)
def test_demo_squadv2(
    device,
    use_program_cache,
    model_name,
    bert,
    batch_size,
    sequence_size,
    n_iterations,
):
    disable_persistent_kernel_cache()

    return run_roberta_question_and_answering_inference_squad_v2(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=batch_size,
        sequence_size=sequence_size,
        bert=bert,
        n_iterations=n_iterations,
    )
