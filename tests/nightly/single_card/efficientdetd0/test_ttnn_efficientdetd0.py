# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.efficientdetd0.tests.pcc.test_efficient_det import test_efficient_det


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch,channels,height,width,use_torch_maxpool",
    [
        (1, 3, 512, 512, True),
    ],
)
def test_efficientdetd0(device, batch, channels, height, width, use_torch_maxpool, model_location_generator):
    test_efficient_det(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        use_torch_maxpool=use_torch_maxpool,
        device=device,
        model_location_generator=model_location_generator,
    )
