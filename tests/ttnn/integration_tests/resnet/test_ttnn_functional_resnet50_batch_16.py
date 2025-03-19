# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.integration_tests.resnet.test_ttnn_functional_resnet50 import run_resnet_50


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_resnet_batch_16(
    device,
    use_program_cache,
    model_location_generator,
):
    run_resnet_50(
        device,
        use_program_cache,
        16,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        False,
        model_location_generator,
    )
