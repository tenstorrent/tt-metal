# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from transformers import AutoTokenizer

from models.experimental.t5.tt.t5_for_conditional_generation import (
    t5_small_for_conditional_generation,
    t5_base_for_conditional_generation,
    flan_t5_small_for_conditional_generation,
)
from models.generation_utils import run_generate
from models.utility_functions import comp_pcc


def run_T5ForConditionalGeneration(device, model_constructor, model_name):
    input_sentance = "translate English to German: The house is wonderful."
    if model_name == "t5-small":
        correct_output = "Das Haus ist wunderbar."
    elif model_name == "google/flan-t5-small":
        correct_output = "Das Haus ist schön."
    elif model_name == "t5-base":
        correct_output = "Das Haus ist wunderbar."

    # input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    # if model_name == "t5-small":
    #     correct_output = "QuillBot's Summarizer wants to change how you read. instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    # elif model_name == "google/flan-t5-small":
    #     correct_output = "QuillBot's Summarizer is a free eBook that lets you read your documents."
    # elif model_name == "t5-base":
    #     correct_output = "QuillBot's Summarizer is a quick and easy way to read documents. instead of reading through documents, you can get a short annotated summary."

    # input_sentance = "translate English to French: Welcome to NYC"
    # if model_name == "t5-small":
    #     correct_output = "Bienvenue à NYC"
    # elif model_name == "google/flan-t5-small":
    #     correct_output = "Accueil à NCT"
    # elif model_name == "t5-base":
    #     correct_output = "Bienvenue à New York"

    # input_sentance = "The <extra_id_0> walks in <extra_id_1> park"
    # if model_name == "t5-small":
    #     correct_output = "park offers the park."
    # elif model_name == "google/flan-t5-small":
    #     correct_output = "a swansea swansea swansea swansea swansea swansea swansea swansea swansea swansea s"
    # elif model_name == "t5-base":
    #     correct_output = "park is a short walk from the park. There are the park is a short walk from the park. There are park has the park has the the park is a short walk from the park. There are the park has park has the"

    # input_sentance = "summarize: I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder"
    # if model_name == "t5-small":
    #     correct_output = "i'm sitting here in a boring room. I'm wasting my time I got nothing to do. I wonder if nothing ever happens."
    # elif model_name == "google/flan-t5-small":
    #     correct_output = "I'm wasting my time."
    # elif model_name == "t5-base":
    #     correct_output = "bob greene: it's another rainy Sunday afternoon. he's wasting his time. he says he's hanging around waiting for you. but nothing ever happens. greene: i wonder if he'll ever get to see you again. he"

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)

    pt_output_sentance = run_generate(
        input_sentance,
        tokenizer,
        model_constructor,
        device,
        run_tt_model=False,
        log=True,
        comp_pcc=comp_pcc,
    )

    tt_output_sentance = run_generate(
        input_sentance,
        tokenizer,
        model_constructor,
        device,
        run_tt_model=True,
        log=True,
        comp_pcc=comp_pcc,
    )

    logger.info(f"Pt Decoded output: {pt_output_sentance}")
    logger.info(f"Tt Decoded output: {tt_output_sentance}")

    assert tt_output_sentance == correct_output


def test_T5ForConditionalGeneration_t5_small(device):
    run_T5ForConditionalGeneration(device, t5_small_for_conditional_generation, "t5-small")


def test_T5ForConditionalGeneration_flan_t5_small(device):
    run_T5ForConditionalGeneration(device, flan_t5_small_for_conditional_generation, "google/flan-t5-small")


def test_T5ForConditionalGeneration_t5_base(device):
    run_T5ForConditionalGeneration(device, t5_base_for_conditional_generation, "t5-base")
