# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.yolov4.tests.pcc.test_ttnn_yolov4 import run_yolov4


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4(device, reset_seeds, model_location_generator):
    run_yolov4(
        device,
        model_location_generator,
        True,
        (640, 640),
    )
