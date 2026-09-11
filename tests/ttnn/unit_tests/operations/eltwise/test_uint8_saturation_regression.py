# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

def test_quantize_uint8_lower_bound_saturation(device):
    input_data = torch.tensor([-10.0, -5.0, -0.01, 0.0, 5.0, 10.0, 300.0], dtype=torch.float32)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.quantize(
        input_tensor,
        scale=1.0,
        zero_point=0,
        axis=None,
        output_dtype=ttnn.uint8,
    )
    output_torch = ttnn.to_torch(output_tensor)

    # Acceptance criteria: values below 0 must produce 0, not wrap or return magnitude
    assert output_torch[0] == 0
    assert output_torch[1] == 0
    assert output_torch[2] == 0
    assert output_torch[3] == 0
    assert output_torch[4] == 5
    assert output_torch[5] == 10
    assert output_torch[6] == 255

def test_requantize_uint8_lower_bound_saturation(device):
    input_data = torch.tensor([-20, -5, 0, 50, 300], dtype=torch.int32)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.requantize(
        input_tensor,
        in_scale=1.0,
        in_zero_point=0,
        out_scale=1.0,
        out_zero_point=0,
        axis=None,
        output_dtype=ttnn.uint8,
    )
    output_torch = ttnn.to_torch(output_tensor)

    assert output_torch[0] == 0
    assert output_torch[1] == 0
    assert output_torch[2] == 0
    assert output_torch[3] == 50
    assert output_torch[4] == 255
