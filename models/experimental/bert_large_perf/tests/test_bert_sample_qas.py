# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline


import time
import random
import json
import ttnn
from models.experimental.bert_large_perf.tt.embeddings import PytorchEmbeddings
from models.experimental.bert_large_perf.tt.bert_encoder import TtBertEncoder
from models.experimental.bert_large_perf.fused_ops.linear import Linear
from models.experimental.bert_large_perf.fused_ops.layernorm import (
    create_var_scaler,
)
from tt_lib.utils import pad_activation, pad_weight
from models.utility_functions import enable_persistent_kernel_cache
from models.utility_functions import profiler
from models.utility_functions import disable_persistent_kernel_cache


class DataSampler:
    def __init__(self, file_name):
        self.data = {}
        try:
            with open(file_name) as json_file:
                self.data = json.load(json_file)
        except FileNotFoundError:
            logger.warning("File not found")

        self.file_name = file_name

    def read(self):
        titles = []

        for topic in self.data["data"]:
            titles.append(topic["title"])

        selection = random.choice(titles)

        for topic in self.data["data"]:
            if topic["title"] != selection:
                continue

            # select paragraph
            total_paragraphs = len(topic["paragraphs"])
            selected_paragraph = random.randint(0, total_paragraphs - 1)
            paragraph = topic["paragraphs"][selected_paragraph]

            # select question
            total_questions = len(paragraph["qas"])
            selected_question = random.randint(0, total_questions - 1)
            qas = paragraph["qas"][selected_question]
            question = qas["question"]

            # get all related answers
            answers = []

            if len(qas["answers"]) == 0:
                for answer in qas["plausible_answers"]:
                    answers.append(answer["text"])
            else:
                for answer in qas["answers"]:
                    answers.append(answer["text"])

            # get context
            context = paragraph["context"]

        return {"context": context, "question": question, "answers": answers}

    def readn(self, num_samples):
        inputs = []

        for i in range(num_samples):
            sample = self.read()
            inputs.append(sample)

        return inputs


def sample_bert_input(
    hugging_face_reference_model,
    tokenizer,
    seq_len,
    attention_mask,
    token_type_ids,
    qas_sample,
    num_samples,
):
    samples = []

    for i in range(num_samples):
        input_qas = qas_sample.read()
        context = [input_qas["context"]]
        question = [input_qas["question"]]

        bert_input = tokenizer.batch_encode_plus(
            zip(question, context),
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        nlp = pipeline(
            "question-answering",
            model=hugging_face_reference_model,
            tokenizer=tokenizer,
        )
        pl_answer = nlp(question=question[0], context=context[0])
        preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
        preprocess_params["max_seq_len"] = seq_len
        input_q = {"context": context[0], "question": question[0]}
        examples = nlp._args_parser(input_q)
        model_input = next(nlp.preprocess(examples[0], **preprocess_params))

        single_input = {
            "data": (
                model_input["input_ids"],
                model_input["attention_mask"] if attention_mask else None,
                model_input["token_type_ids"] if token_type_ids else None,
            ),
            "example": model_input["example"],
            "inputs": model_input,
        }

        bert_input = {}
        bert_input["input_ids"] = single_input["data"][0]
        bert_input["attention_mask"] = single_input["data"][1]
        bert_input["token_type_ids"] = single_input["data"][2]

        sample = {}
        sample["bert_input"] = bert_input
        sample["single_input"] = single_input
        sample["nlp"] = nlp
        sample["postprocess_params"] = postprocess_params
        sample["pl_answer"] = pl_answer
        sample["context"] = context
        sample["question"] = question
        sample["answers"] = input_qas["answers"]

        samples.append(sample)

    return samples


class TtBertForQuestionAnswering(torch.nn.Module):
    def __init__(self, config, hugging_face_reference_model, seq_len, device):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        # Constant prop -> create_var_scaler
        var_scaler = create_var_scaler(seq_len, config.hidden_size, config.layer_norm_eps, device)

        self.hidden_states_list = []
        self.tt_attention_mask_list = []

        # So far on CPU until we add embeddings support on device
        self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.get_extended_attention_mask = hugging_face_reference_model.get_extended_attention_mask
        self.encoders = torch.nn.ModuleList(
            [
                TtBertEncoder(config, encoder_idx, state_dict, var_scaler, device)
                for encoder_idx in range(config.num_hidden_layers)
            ]
        )

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        weight = pad_weight(state_dict["qa_outputs.weight"])
        weight = (
            ttnn.Tensor(
                weight.reshape(-1).tolist(),
                weight.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        bias = pad_weight(state_dict["qa_outputs.bias"])
        bias = (
            ttnn.Tensor(
                bias.reshape(-1).tolist(),
                bias.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

        # QA linear
        self.qa_linear = Linear(hidden_size, 32, weight, bias, device)
        self.device = device

    def forward(self, samples):
        for sample in samples:
            profiler.start("_calc_embeddings")

            input_ids = sample["bert_input"]["input_ids"]
            attention_mask = sample["bert_input"]["attention_mask"]
            token_type_ids = sample["bert_input"]["token_type_ids"]

            embeddings = self.embeddings(input_ids, token_type_ids)

            if attention_mask is not None:
                extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
                extended_attention_mask = torch.clamp(
                    extended_attention_mask, -100000
                )  # Limit neg value that goes into exp
                extended_attention_mask = pad_activation(extended_attention_mask)
                tt_attention_mask = ttnn.Tensor(
                    extended_attention_mask.reshape(-1).tolist(),
                    extended_attention_mask.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                ).to(ttnn.TILE_LAYOUT)
                tt_attention_mask = tt_attention_mask.to(self.device)
            else:
                tt_attention_mask = attention_mask

            # Add to list mask
            self.tt_attention_mask_list.append(tt_attention_mask)

            # Convert to ll buda tensor
            pad_embeddings = pad_activation(embeddings)
            tt_embeddings = ttnn.Tensor(
                pad_embeddings.reshape(-1).tolist(),
                (
                    pad_embeddings.shape[0],
                    1,
                    pad_embeddings.shape[-2],
                    pad_embeddings.shape[-1],
                ),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
            tt_embeddings = tt_embeddings.to(self.device)
            hidden_states = tt_embeddings  # pad_embeddings #

            self.hidden_states_list.append(hidden_states)
            profiler.end("_calc_embeddings")

        tt_out_list = []

        for i in range(len(self.hidden_states_list)):
            logger.debug(f"Running BERT model for sample {i}")
            profiler.start("_run_encoders")

            hidden_states = self.hidden_states_list[i]
            attention_mask = self.tt_attention_mask_list[i]

            for encoder in self.encoders:
                profiler.start("__one_encoder")
                hidden_states = encoder(hidden_states, attention_mask)
                profiler.end("__one_encoder")

            profiler.end("_run_encoders")

            profiler.start("_qa_linear")
            res = self.qa_linear(hidden_states)
            profiler.end("_qa_linear")

            tt_out_list.append(res)

        return tt_out_list


def run_bert_question_and_answering_inference(
    model_version,
    batch,
    seq_len,
    attention_mask,
    token_type_ids,
    pcc,
    model_location_generator,
    qas_sample,
    num_samples,
    device,
):
    torch.manual_seed(1234)

    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_bert_model = TtBertForQuestionAnswering(
        hugging_face_reference_model.config,
        hugging_face_reference_model,
        seq_len,
        device,
    )

    profiler.start("processing_of_input")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Sampling random context+question pairs")
    samples = sample_bert_input(
        hugging_face_reference_model,
        tokenizer,
        seq_len,
        attention_mask,
        token_type_ids,
        qas_sample,
        num_samples,
    )
    profiler.end("processing_of_input")

    logger.info(f"Running BERT model")
    profiler.start("whole_model")
    tt_out_list = tt_bert_model(samples)
    profiler.end("whole_model", num_samples)

    profiler.start("processing_output_to_string")

    for i in range(len(tt_out_list)):
        single_input = samples[i]["single_input"]
        nlp = samples[i]["nlp"]
        postprocess_params = samples[i]["postprocess_params"]
        tt_out = tt_out_list[i]
        context = samples[i]["context"]
        question = samples[i]["question"]
        answers = samples[i]["answers"]

        tt_out = tt_out.cpu()
        tt_untilized_output = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch().reshape(batch, 1, seq_len, -1)

        tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
        tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)

        tt_res = {
            "start": tt_start_logits,
            "end": tt_end_logits,
            "example": single_input["example"],
            **single_input["inputs"],
        }

        tt_answer = nlp.postprocess([tt_res], **postprocess_params)["answer"]

        logger.info(f"Context: {context}")
        logger.info(f"Question: {question}")
        logger.info(f"Answer from GS: '{tt_answer}'")
        logger.info(f"All valid answers: {answers}\n")

    profiler.end("processing_output_to_string")

    profiler.print()


def test_bert_sample_qas(device, model_location_generator):
    model_version = "phiyodr/bert-large-finetuned-squad2"
    batch = 1
    seq_len = 384
    attention_mask = True
    token_type_ids = True
    pcc = 0.98
    num_samples = 10

    qas_sample = DataSampler("./tests/models/bert_large_perf/dev-v2.0.json")

    logger.warning("This test uses binary and compile cache. The cache needs to be filled before running this test.")

    run_bert_question_and_answering_inference(
        model_version,
        batch,
        seq_len,
        attention_mask,
        token_type_ids,
        pcc,
        model_location_generator,
        qas_sample,
        num_samples,
        device,
    )
