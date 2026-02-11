# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.wormhole.bark.demo.demo import run_demo

@pytest.fixture
def device():
    import ttnn
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

@pytest.mark.parametrize(
    "text, output_file",
    [
        ("Hello world [laughs]", "models/demos/wormhole/bark/demo/output_test.wav"),
    ],
)
def test_bark_demo(device, text, output_file):
    run_demo(text, output_file, device)
