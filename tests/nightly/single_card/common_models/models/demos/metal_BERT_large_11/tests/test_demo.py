# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.metal_BERT_large_11.demo.demo import test_demo as demo_json
from models.demos.metal_BERT_large_11.demo.demo import test_demo_squadv2 as demo_squadv2
import pytest
from loguru import logger
from models.utility_functions import is_e75, is_wormhole_b0, skip_for_grayskull, is_blackhole


@skip_for_grayskull()
@pytest.mark.parametrize("batch", (7,), ids=["batch_7"])
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/metal_BERT_large_11/demo/input_data.json"),),
    ids=["default_input"],
)
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="#7525: hangs on wh b0")
def test_demo_batch_7(batch, input_path, model_location_generator, device, use_program_cache):
    if is_e75(device):
        pytest.skip(f"Bert large 11 is not supported on E75")

    expected_answers = {
        0: "scientific archaeology",
        1: "Richard I",
        2: "males",
        3: "The Huguenots adapted quickly and often married outside their immediate French communities,",
        4: "biostratigraphers",
        5: "chemotaxis,",
        6: "1992,",
        7: "statocyst,",
        8: "color field paintings",
        9: "paranoia",
        10: "six months earlier.",
        11: "large head and neck",
    }
    NUM_RUNS = 10
    measurements, answers = demo_json(batch, input_path, NUM_RUNS, model_location_generator, device, use_program_cache)
    logger.info(measurements)

    logger.info(answers)

    for i in range(batch):
        assert expected_answers[i] == answers[i]


@pytest.mark.parametrize("batch", (12,), ids=["batch_12"])
@pytest.mark.parametrize(
    "input_path",
    (("models/demos/metal_BERT_large_11/demo/input_data.json"),),
    ids=["default_input"],
)
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="#7525: hangs on wh b0")
def test_demo_batch_12(batch, input_path, model_location_generator, device, use_program_cache):
    if is_e75(device):
        pytest.skip(f"Bert large 11 is not supported on E75")

    expected_answers = {
        0: "scientific archaeology",
        1: "Richard I",
        2: "males",
        3: "married outside their immediate French communities,",
        4: "biostratigraphers",
        5: "chemotaxis,",
        6: "1992,",
        7: "statocyst,",
        8: "color field paintings",
        9: "paranoia",
        10: "six months earlier.",
        11: "large head and neck",
    }
    NUM_RUNS = 10
    measurements, answers = demo_json(batch, input_path, NUM_RUNS, model_location_generator, device, use_program_cache)
    logger.info(measurements)

    logger.info(answers)

    for i in range(batch):
        assert expected_answers[i] == answers[i]


@skip_for_grayskull()
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="#7525: hangs on wh b0")
@pytest.mark.parametrize(
    "batch, exact, f1",
    (
        (
            7,
            78.57,
            84.37,
        ),
    ),
    ids=["batch_7"],
)
def test_demo_squadv2_batch_7(batch, exact, f1, model_location_generator, device, use_program_cache):
    loop_count = 10
    evals = demo_squadv2(model_location_generator, device, use_program_cache, batch, loop_count)

    assert evals["exact"] >= exact
    assert evals["f1"] >= f1


@pytest.mark.parametrize(
    "batch, exact, f1",
    (
        (
            12,
            80,
            86,
        ),
    ),
    ids=["batch_12"],
)
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="#7525: hangs on wh b0")
def test_demo_squadv2_batch_12(batch, exact, f1, model_location_generator, device, use_program_cache):
    loop_count = 10
    evals = demo_squadv2(model_location_generator, device, use_program_cache, batch, loop_count)

    assert evals["exact"] >= exact
    assert evals["f1"] >= f1
