# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.experimental.functional_whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper
from models.experimental.functional_whisper.demo.demo import test_demo_for_conditional_generation as demo
from models.experimental.functional_whisper.demo.demo import (
    test_demo_for_conditional_generation_dataset as demo_dataset,
)


@pytest.mark.parametrize(
    "input_path",
    (("models/experimental/functional_whisper/demo/dataset/conditional_generation"),),
    ids=["default_input"],
)
@pytest.mark.parametrize(
    "batch_size",
    (10,),
    ids=["batch_10"],
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
def test_demo_batch_10(input_path, ttnn_model, device, use_program_cache, reset_seeds, batch_size):
    expected_answers = {
        0: " As soon as you",
        1: " Some festivals have special",
        2: " The original population hasn",
        3: " Although three people ",
        4: " Soon, officers",
        5: " Water is spilling over",
        6: " Naturalist and Philos",
        7: " With only 18 metals",
        8: " Scientists say the explosion",
        9: " According to police the",
    }
    NUM_RUNS = 5
    measurements, answers = demo(input_path, ttnn_model, device, use_program_cache, reset_seeds, batch_size, NUM_RUNS)

    logger.info(measurements)
    logger.info(answers)


@pytest.mark.parametrize(
    "batch_size, wer",
    (
        (
            7,
            0.86,
        ),
    ),
    ids=["batch_7"],
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
def test_demo_squadv2_batch_7(ttnn_model, device, reset_seeds, batch_size, wer, use_program_cache):
    loop_count = 5
    evals = demo_dataset(
        ttnn_model, device, use_program_cache, reset_seeds, batch_size, n_iterations=1, max_tokens=loop_count
    )
    assert evals <= wer
