# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import tt_lib
from loguru import logger
from transformers import AutoTokenizer

from models.t5.tt.t5_for_conditional_generation import (
    t5_small_for_conditional_generation,
    t5_base_for_conditional_generation,
    flan_t5_small_for_conditional_generation,
)
from models.generation_utils import run_generate
from models.utility_functions import comp_pcc


def run_T5ForConditionalGeneration(model_constructor, model_name):
    input_sentance = "translate English to German: The house is wonderful."
    if model_name == "t5-small":
        correct_output = "Das Haus ist wunderbar."
    elif model_name == "google/flan-t5-small":
        correct_output = "Das Haus ist schön."
    elif model_name == "t5-base":
        correct_output = "Das Haus ist wunderbar."

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)

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

    tt_lib.device.CloseDevice(device)
    assert tt_output_sentance == correct_output


@pytest.mark.parametrize(
    "pcc, model_name",
    ((0.99, "t5-small"),),
)
def test_T5ForConditionalGeneration_t5_small(pcc, model_name):
    run_T5ForConditionalGeneration(t5_small_for_conditional_generation, model_name)


@pytest.mark.parametrize(
    "pcc, model_name",
    ((0.99, "google/flan-t5-small"),),
)
def test_T5ForConditionalGeneration_flan_t5_small(pcc, model_name):
    run_T5ForConditionalGeneration(flan_t5_small_for_conditional_generation, model_name)


@pytest.mark.parametrize(
    "pcc, model_name",
    ((0.99, "t5-base"),),
)
def test_T5ForConditionalGeneration_t5_base(pcc, model_name):
    run_T5ForConditionalGeneration(t5_base_for_conditional_generation, model_name)
