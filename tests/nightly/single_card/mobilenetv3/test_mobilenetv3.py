# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger

from models.experimental.mobileNetV3.tests.pcc.test_mobilenetv3 import MobilenetV3TestInfra


@pytest.mark.parametrize("device_params", [{"l1_small_size": 12288}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 224, 224),
    ],
)
def test_MobilenetV3(device, batch_size, input_channels, height, width):
    MobilenetV3TestInfra(
        device=device,
        batch_size=batch_size,
        input_channels=input_channels,
        height=height,
        width=width,
    )
