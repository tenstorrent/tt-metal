# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.integration_tests.resnet.test_ttnn_functional_resnet50 import resnet_batch_16_lofi_pretrained_false


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_resnet_batch_16(
    device,
    use_program_cache,
    model_location_generator,
):
    resnet_batch_16_lofi_pretrained_false(
        device,
        use_program_cache,
        model_location_generator,
    )
