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
    enable_persistent_kernel_cache,
    profiler,
)

from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from models.demos.metal_BERT_large_11.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
    skip_unsupported_config,
)
from models.demos.metal_BERT_large_11.tt.bert_model import TtBertBatchDram

from models.datasets.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch

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


def run_bert_question_and_answering_inference_squadv2(
    model_version,
    batch,
    seq_len,
    return_attention_mask,
    return_token_type_ids,
    model_config,
    tt_cache_path,
    model_location_generator,
    device,
    loop_count,
):
    BATCH_SIZE = batch
    comments = "Large"
    seq_len = 384
    real_input = True
    attention_mask = True
    token_type_ids = True

    # set up huggingface model - TT model will use weights from this model
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    nlp = pipeline(
        "question-answering",
        model=hugging_face_reference_model,
        tokenizer=tokenizer,
    )

    context = BATCH_SIZE * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = BATCH_SIZE * ["What discipline did Winkelmann create?"]

    inputs = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        # truncation=True,
        truncation="only_second",
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt",
    )

    # SQUaD-v2 - 1000 samples
    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, seq_len, attention_mask, token_type_ids, BATCH_SIZE)
    squad_metric = evaluate.load("squad_v2")

    tt_bert_model = TtBertBatchDram(
        hugging_face_reference_model.config,
        hugging_face_reference_model,
        device,
        model_config,
        tt_cache_path,
    )
    with torch.no_grad():
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            if i < loop_count:
                logger.info(f"BATCH: {i}")
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]
                if curr_batch_size < BATCH_SIZE:
                    batch_data["input_ids"] = torch.nn.functional.pad(
                        batch_data["input_ids"], (0, 0, 0, BATCH_SIZE - curr_batch_size)
                    )
                    batch_data["attention_mask"] = torch.nn.functional.pad(
                        batch_data["attention_mask"], (0, 0, 0, BATCH_SIZE - curr_batch_size)
                    )
                    batch_data["token_type_ids"] = torch.nn.functional.pad(
                        batch_data["token_type_ids"], (0, 0, 0, BATCH_SIZE - curr_batch_size)
                    )
                cpu_output = hugging_face_reference_model(**batch_data)
                tt_attention_mask = tt_bert_model.model_attention_mask(**batch_data)
                tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**batch_data)

                tt_attention_mask = tt_attention_mask.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
                tt_embedding_inputs = {
                    key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"])
                    for (key, value) in tt_embedding_inputs.items()
                }
                tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)

                # tt_batch = tt_bert_model.model_preprocessing(**batch_data)
                # tt_output = tt_bert_model(*tt_batch)
                tt_output = tt_bert_model(tt_embedding, tt_attention_mask).cpu()

                tt_output = (
                    tt_output.to(ttnn.ROW_MAJOR_LAYOUT).to_torch().reshape(BATCH_SIZE, 1, seq_len, -1).to(torch.float32)
                )
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
                del tt_attention_mask
                del tt_embedding_inputs
                del tt_embedding
                del tt_output
            i += 1
        eval_score = squad_metric.compute(predictions=pred_labels, references=true_labels)
        cpu_eval_score = squad_metric.compute(predictions=cpu_pred_labels, references=true_labels)
        logger.info(f"\tTT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")
        logger.info(f"\tCPU_Eval: exact: {cpu_eval_score['exact']} -- F1:  {cpu_eval_score['f1']}")

        ttnn.synchronize_device(device)

        return eval_score


def run_bert_question_and_answering_inference(
    model_version,
    batch,
    seq_len,
    return_attention_mask,
    return_token_type_ids,
    model_config,
    tt_cache_path,
    NUM_RUNS,
    input_path,
    model_location_generator,
    device,
):
    torch.manual_seed(1234)

    # set up huggingface model - TT model will use weights from this model
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
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
        tt_cache_path,
    )
    profiler.end(f"move_weights")

    # create inputs for TT model
    profiler.start(f"processing_input_two")
    profiler.start("attention_mask_preprocessing")
    tt_attention_mask = tt_bert_model.model_attention_mask(**bert_input)
    profiler.end("attention_mask_preprocessing")

    profiler.start("embedding_input_preprocessing")
    tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**bert_input)
    profiler.end("embedding_input_preprocessing")
    profiler.end(f"processing_input_two")

    # profiler.start(f"processing_input_two")
    # tt_bert_input = tt_bert_model.model_preprocessing(**bert_input)
    # profiler.end(f"processing_input_two")

    ##### Run TT Model to Fill Cache Start
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_attention_mask = tt_attention_mask.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
    tt_embedding_inputs = {
        key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"]) for (key, value) in tt_embedding_inputs.items()
    }
    tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)
    tt_out = tt_bert_model(tt_embedding, tt_attention_mask).cpu()
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out

    # Recreate inputs since activations were deallocated
    tt_attention_mask = tt_bert_model.model_attention_mask(**bert_input)
    tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**bert_input)

    profiler.enable()
    enable_persistent_kernel_cache()

    ##### Run Forward on TT Model Start
    profiler.start(f"model_run_for_inference")
    tt_attention_mask = tt_attention_mask.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
    tt_embedding_inputs = {
        key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"]) for (key, value) in tt_embedding_inputs.items()
    }
    tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)
    tt_out = tt_bert_model(tt_embedding, tt_attention_mask).cpu()
    profiler.end(f"model_run_for_inference")

    # running in a loop
    for i in range(NUM_RUNS):
        tt_attention_mask = tt_bert_model.model_attention_mask(**bert_input)
        tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**bert_input)
        tt_attention_mask = tt_attention_mask.to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
        tt_embedding_inputs = {
            key: value.to(device, model_config["INPUT_EMBEDDINGS_MEMCFG"])
            for (key, value) in tt_embedding_inputs.items()
        }
        tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)
        _tt_out = tt_bert_model(tt_embedding, tt_attention_mask).cpu()

    ##### Output Postprocessing Start
    profiler.start("processing_output_to_string")

    # convert TT Tensor returned from GS device to Torch tensor
    tt_untilized_output = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch().reshape(batch, 1, seq_len, -1).to(torch.float32)
    # extract logits for start and end of answer string
    tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)
    model_answers = {}
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
        model_answers[i] = tt_answer["answer"]

    profiler.end("processing_output_to_string")
    ##### Output Postprocessing End
    SINGLE_RUN = 1
    measurements = {
        "preprocessing": profiler.get("processing_input_one") + profiler.get("processing_input_two"),
        "moving_weights_to_device": profiler.get("move_weights"),
        "compile": profiler.get("first_model_run_with_compile")
        - (profiler.get("model_run_for_inference") / SINGLE_RUN),
        f"inference_for_single_run_batch_{batch}_without_cache": profiler.get("first_model_run_with_compile"),
        f"inference_for_{SINGLE_RUN}_run_batch_{batch}_without_cache": profiler.get("model_run_for_inference"),
        "inference_throughput": (SINGLE_RUN * batch) / profiler.get("model_run_for_inference"),
        "post_processing": profiler.get("processing_output_to_string"),
    }

    logger.info(f"pre processing duration: {measurements['preprocessing']} s")
    logger.info(f"moving weights to device duration: {measurements['moving_weights_to_device']} s")
    logger.info(f"compile time: {measurements['compile']} s")
    logger.info(
        f"inference time for single run of model with batch size {batch} without using cache: {measurements[f'inference_for_single_run_batch_{batch}_without_cache']} s"
    )
    logger.info(
        f"inference time for {SINGLE_RUN} run(s) of model with batch size {batch} and using cache: {measurements[f'inference_for_{SINGLE_RUN}_run_batch_{batch}_without_cache']} s"
    )
    logger.info(f"inference throughput: {measurements['inference_throughput'] } inputs/s")
    logger.info(f"post processing time: {measurements['post_processing']} s")

    del tt_out
    return measurements, model_answers


@pytest.mark.parametrize("batch", (7, 8, 12), ids=["batch_7", "batch_8", "batch_12"])
@pytest.mark.parametrize(
    "input_path, NUM_RUNS",
    (("models/demos/metal_BERT_large_11/demo/input_data.json", 1),),
)
def test_demo(
    batch,
    input_path,
    NUM_RUNS,
    model_location_generator,
    device,
    use_program_cache,
):
    model_config_str = "BFLOAT8_B-SHARDED"
    skip_unsupported_config(device, model_config_str, batch)
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bert_question_and_answering_inference(
        model_version="phiyodr/bert-large-finetuned-squad2",
        batch=batch,
        seq_len=384,
        return_attention_mask=True,
        return_token_type_ids=True,
        model_config=get_model_config(batch, device.compute_with_storage_grid_size(), model_config_str),
        tt_cache_path=get_tt_cache_path("phiyodr/bert-large-finetuned-squad2"),
        NUM_RUNS=NUM_RUNS,
        input_path=input_path,
        model_location_generator=model_location_generator,
        device=device,
    )


@pytest.mark.parametrize("batch", (7, 8, 12), ids=["batch_7", "batch_8", "batch_12"])
@pytest.mark.parametrize(
    "loop_count",
    ((20),),
)
def test_demo_squadv2(model_location_generator, device, use_program_cache, batch, loop_count):
    model_config_str = "BFLOAT8_B-SHARDED"
    skip_unsupported_config(device, model_config_str, batch)
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bert_question_and_answering_inference_squadv2(
        model_version="phiyodr/bert-large-finetuned-squad2",
        batch=batch,
        seq_len=384,
        return_attention_mask=True,
        return_token_type_ids=True,
        model_config=get_model_config(batch, device.compute_with_storage_grid_size(), model_config_str),
        tt_cache_path=get_tt_cache_path("phiyodr/bert-large-finetuned-squad2"),
        model_location_generator=model_location_generator,
        device=device,
        loop_count=loop_count,
    )
