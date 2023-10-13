# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.metal_BERT_large_15.demo.demo import test_demo as demo
import pytest
from loguru import logger


@pytest.mark.parametrize(
    "input_path",
    (("models/demos/metal_BERT_large_15/demo/input_data.json"),),
    ids=["default_input"],
)
def test_demo(
    input_path,
    model_location_generator,
    device,
    use_program_cache
):
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
    measurements, answers = demo(input_path, model_location_generator, device, use_program_cache)
    logger.info(measurements)

    assert measurements["preprocessing"] < 0.12
    assert measurements["moving_weights_to_device"] < 40
    assert measurements["compile"] < 10.5
    assert measurements["inference_for_single_run_batch_8_without_cache"] < 10.5
    assert measurements["inference_for_1_runs_batch_8_without_cache"] < 0.42
    assert measurements["inference_throughput"] > 105
    assert measurements["post_processing"] < 0.035

    logger.info(answers)

    for key, value in expected_answers.items():
        assert value == answers[key]
