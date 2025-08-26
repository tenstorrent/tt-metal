# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov8x.common import YOLOV8X_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8X_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 1, 25600, 80)))], ids=["input_tensor1"])
def test_concat(device, input_tensor, model_location_generator):
    input_a = ttnn.from_torch(
        input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_b = ttnn.from_torch(
        input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_c = ttnn.from_torch(
        input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_d = ttnn.from_torch(
        input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.concat([input_a, input_b, input_c, input_d], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
