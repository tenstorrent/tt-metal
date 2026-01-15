# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.retinanet.tests.pcc.test_retinanet import RetinaNetTestInfra, model_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,in_channels,height,width",
    [
        (1, 3, 512, 512),
    ],
)
def test_retinanet(device, batch_size, in_channels, height, width):
    RetinaNetTestInfra(
        device=device,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        model_config=model_config,
        name="retinanet",
    )
