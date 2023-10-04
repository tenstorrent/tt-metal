# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch

from loguru import logger

import tt_lib
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler
)

from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from models.experimental.metal_BERT_large_15.tt.model_config import get_model_config
from models.experimental.metal_BERT_large_15.tt.bert_model import TtBertBatchDram


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

def run_bert_question_and_answering_inference(
    model_version,
    batch,
    seq_len,
    return_attention_mask,
    return_token_type_ids,
    model_config,
    NUM_RUNS,
    input_path,
    model_location_generator,
    device,
):
    torch.manual_seed(1234)

    # set up huggingface model - TT model will use weights from this model
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    nlp = pipeline(
        "question-answering",
        model=hugging_face_reference_model,
        tokenizer=tokenizer,
    )

    # load context+question from provided input file
    context, question = load_inputs(input_path, batch)

    # Prepare preprocessed_inputs which will be used with nlp.postprocess to extract string
    # answers from context after TT models produces start and end logits

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
    preprocess_params["max_seq_len"] = seq_len
    inputs = nlp._args_parser({"context": context, "question": question})
    preprocessed_inputs = []
    for i in range(batch):
        model_input = next(nlp.preprocess(inputs[0][i], **preprocess_params))
        single_input = {
            "example": model_input["example"],
            "inputs": model_input,
        }
        preprocessed_inputs.append(single_input)

    # encode input context+question strings
    profiler.start(f"processing_input_one")
    bert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        return_tensors="pt",
    )
    profiler.end(f"processing_input_one")


    ##### Create TT Model Start
    # this will move weights to device
    profiler.start(f"move_weights")
    tt_bert_model = TtBertBatchDram(
        hugging_face_reference_model.config,
        hugging_face_reference_model,
        device,
        model_config,
    )
    profiler.end(f"move_weights")


    # create inputs for TT model
    profiler.start(f"processing_input_two")
    tt_bert_input = tt_bert_model.model_preprocessing(**bert_input)
    profiler.end(f"processing_input_two")

    ##### Run TT Model to Fill Cache Start
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_out = tt_bert_model(1, *tt_bert_input)
    tt_lib.device.Synchronize()
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out

    # Recreate inputs since activations were deallocated
    tt_bert_input = tt_bert_model.model_preprocessing(**bert_input)
    profiler.enable()
    enable_persistent_kernel_cache()

    ##### Run Forward on TT Model Start
    profiler.start(f"model_run_for_inference")
    tt_out = tt_bert_model(NUM_RUNS, *tt_bert_input)
    tt_lib.device.Synchronize()
    profiler.end(f"model_run_for_inference")



    ##### Output Postprocessing Start
    profiler.start("processing_output_to_string")

    # convert TT Tensor returned from GS device to Torch tensor
    tt_untilized_output = (
        tt_out.cpu()
        .to(tt_lib.tensor.Layout.ROW_MAJOR)
        .to_torch()
        .reshape(batch, 1, seq_len, -1)
        .to(torch.float32)
    )
    # extract logits for start and end of answer string
    tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)

    for i in range(batch):
        tt_res = {
            "start": tt_start_logits[i],
            "end": tt_end_logits[i],
            "example": preprocessed_inputs[i]["example"],
            **preprocessed_inputs[i]["inputs"],
        }
        # get answer string from context based on start and end logits
        tt_answer = nlp.postprocess([tt_res], **postprocess_params)

        # print context, question, and generated answer
        logger.info(f"Input {i}")
        logger.info(f"context: {context[i]}")
        logger.info(f"question: {question[i]}")
        logger.info(f"answer: {tt_answer['answer']}\n")


    profiler.end("processing_output_to_string")
    ##### Output Postprocessing End

    logger.info(f"pre processing duration: {profiler.get('processing_input_one') + profiler.get('processing_input_two')} s")
    logger.info(f"moving weights to device duration: {profiler.get('move_weights')} s")
    logger.info(f"compile time: {profiler.get('first_model_run_with_compile') - (profiler.get('model_run_for_inference') / NUM_RUNS)} s")
    logger.info(f"inference time for single run of model with batch size {batch} without using cache: {profiler.get('first_model_run_with_compile')} s")
    logger.info(f"inference time for {NUM_RUNS} run(s) of model with batch size {batch} and using cache: {profiler.get('model_run_for_inference')} s")
    logger.info(f"inference throughput: {(NUM_RUNS * batch) / profiler.get('model_run_for_inference') } inputs/s")
    logger.info(f"post processing time: {profiler.get('processing_output_to_string')} s")

    del tt_out


def test_demo(
    input_path,
    model_location_generator,
    request,
    device,
):

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/metal_BERT_large_15"
    )

    run_bert_question_and_answering_inference(
        model_version = "phiyodr/bert-large-finetuned-squad2",
        batch = 9,
        seq_len = 384,
        return_attention_mask = True,
        return_token_type_ids = True,
        model_config = get_model_config("BFLOAT16-L1"),
        NUM_RUNS = 1,
        input_path = input_path,
        model_location_generator = model_location_generator,
        device = device,
    )
