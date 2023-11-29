# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.metal_BERT_large_11.demo.demo import test_demo as demo_json
from models.demos.metal_BERT_large_11.demo.demo import test_demo_squadv2 as demo_squadv2
import pytest
from loguru import logger
from models.utility_functions import is_e75


@pytest.mark.parametrize(
    "input_path",
    (("models/demos/metal_BERT_large_11/demo/input_data.json"),),
    ids=["default_input"],
)
def test_demo(input_path, model_location_generator, device, use_program_cache):
    if is_e75(device):
        pytest.skip(f"Bert large 15 is not supported on E75")

    expected_answers = {
        0: "scientific archaeology",
        1: "Richard I of Normandy",
        2: "males",
        3: "The Huguenots adapted quickly and often married outside their immediate French communities,",
        4: "biostratigraphers",
        5: "chemotaxis,",
        6: "1992,",
        7: "statocyst,",
    }
    NUM_RUNS = 1000
    measurements, answers = demo_json(input_path, NUM_RUNS, model_location_generator, device, use_program_cache)
    logger.info(measurements)

    logger.info(answers)

    for key, value in expected_answers.items():
        assert value == answers[key]


def test_demo_squadv2(model_location_generator, device, use_program_cache):
    loop_count = 100
    evals = demo_squadv2(model_location_generator, device, use_program_cache, loop_count)

    assert evals["exact"] > 80
    assert evals["f1"] > 88
