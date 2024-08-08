# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull
from models.demos.grayskull.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.demos.grayskull.functional_bloom.demo.demo_causal_lm import test_demo as demo_cg_json
from models.demos.grayskull.functional_bloom.demo.demo_causal_lm import test_demo_hellaswag as demo_cg_hellaswag
from models.demos.grayskull.functional_bloom.demo.demo_qa import test_demo as demo_qa_json
from models.demos.grayskull.functional_bloom.demo.demo_qa import test_demo_squadv2 as demo_qa_squadv2


@pytest.mark.parametrize(
    "input_path",
    (("models/demos/grayskull/functional_bloom/demo/input_data_causal_lm.json"),),
    ids=["default_input"],
)
@pytest.mark.parametrize(
    "ttnn_model, batch_size",
    ((ttnn_optimized_functional_bloom, 7),),
    ids=["batch_7"],
)
@skip_for_grayskull(reason_str="#10797: OOM")
@skip_for_wormhole_b0()
def test_demo_batch_7_cg(
    input_path, ttnn_model, model_location_generator, device, use_program_cache, batch_size, reset_seeds
):
    expected_answers = {
        0: "A man is sitting on a roof. He is wearing a hat",
        1: "A boy is running down a track. He is a man who",
        2: "A lady walks to a barbell. She is wearing a black",
        3: "Children bring desert out for their family member. The desert is a",
        4: "A cat is sitting in a cat bed. The cat is sitting",
        5: "We see a bottle of face wash. The bottle is a bottle",
        6: "In home pet groomers demonstrate how to make a pet’s",
    }
    NUM_RUNS = 5
    measurements, answers = demo_cg_json(
        input_path, ttnn_model, model_location_generator, device, use_program_cache, batch_size, NUM_RUNS
    )

    logger.info(measurements)
    logger.info(answers)

    for i in range(batch_size):
        assert expected_answers[i] == answers[i]


@pytest.mark.parametrize(
    "ttnn_model, batch_size, ref_accuracy",
    ((ttnn_optimized_functional_bloom, 7, 0.5),),
    ids=["batch_7"],
)
@skip_for_wormhole_b0()
@skip_for_grayskull(reason_str="#10797: OOM")
def test_demo_squadv2_batch_7_cg(
    model_location_generator, ttnn_model, device, use_program_cache, batch_size, ref_accuracy, reset_seeds
):
    loop_count = 2
    NUM_RUNS = 5
    acc = demo_cg_hellaswag(
        model_location_generator, ttnn_model, device, use_program_cache, loop_count, batch_size, NUM_RUNS
    )
    assert acc["accuracy"] >= ref_accuracy


@pytest.mark.parametrize(
    "input_path",
    (("models/demos/grayskull/functional_bloom/demo/input_data_qa.json"),),
    ids=["default_input"],
)
@pytest.mark.parametrize(
    "ttnn_model, batch_size",
    ((ttnn_optimized_functional_bloom, 7),),
    ids=["batch_7"],
)
@skip_for_grayskull(reason_str="#10797: OOM")
@skip_for_wormhole_b0()
def test_demo_batch_7_qa(
    input_path, ttnn_model, model_location_generator, device, use_program_cache, reset_seeds, batch_size
):
    expected_answers = {
        0: "Chopin's performances were",
        1: "The first is the composer",
        2: "The early 20th century.",
        3: "Yes. He was a",
        4: "Beyoncé is a family",
        5: "The archbishop of Cant",
        6: "The city of the Holy",
    }
    NUM_RUNS = 5
    measurements, answers = demo_qa_json(
        input_path, ttnn_model, model_location_generator, device, use_program_cache, reset_seeds, batch_size, NUM_RUNS
    )
    logger.info(measurements)
    logger.info(answers)

    for i in range(batch_size):
        assert expected_answers[i] == answers[i]


@pytest.mark.parametrize(
    "ttnn_model, batch_size, f1",
    ((ttnn_optimized_functional_bloom, 6, 3.72),),
    ids=["batch_6"],
)
@skip_for_grayskull(reason_str="#10797: OOM")
@skip_for_wormhole_b0()
def test_demo_squadv2_batch_6_qa(ttnn_model, device, use_program_cache, reset_seeds, batch_size, f1):
    loop_count = 5
    eval_score = demo_qa_squadv2(
        ttnn_model,
        device,
        use_program_cache,
        reset_seeds,
        batch_size,
        loop_count,
    )
    assert eval_score["f1"] >= f1
