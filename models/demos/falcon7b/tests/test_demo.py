# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.falcon7b.demo.demo import test_demo as demo
import pytest
from loguru import logger
import json

@pytest.mark.parametrize(
    "input_path",
    (("models/demos/falcon7b/demo/input_data.json"),),
    ids=["default_input"],
)
def test_demo(
    input_path,
    model_location_generator,
    device,
    use_program_cache
):

    generated_text, _ = demo(input_path, model_location_generator, device, use_program_cache)

    with open("models/demos/falcon7b/tests/expected_output.json") as handle:
        expected_generated_text = json.loads(handle.read())

    for i, (user_id, output) in enumerate(expected_generated_text.items()):
        assert output == generated_text[i]
