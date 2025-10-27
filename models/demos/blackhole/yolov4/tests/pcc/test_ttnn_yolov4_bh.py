# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.blackhole.yolov4.tests.pcc.test_ttnn_yolov4 import run_yolov4
from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE_BH


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV4_L1_SMALL_SIZE_BH}], indirect=True)
def test_yolov4(device, reset_seeds, model_location_generator):
    run_yolov4(
        device,
        model_location_generator,
        True,
        (640, 640),
    )
