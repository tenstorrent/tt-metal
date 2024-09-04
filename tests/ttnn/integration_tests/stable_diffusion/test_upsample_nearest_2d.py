# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)

from models.utility_functions import torch_random


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_shape", [(2, 1280, 8, 8), (2, 1280, 16, 16), (2, 640, 32, 32)])
@pytest.mark.parametrize("scale_factor", [2])
def test_upsample_nearest2d_512x512(reset_seeds, device, input_shape, scale_factor):
    torch_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = torch.repeat_interleave(torch_tensor, scale_factor, dim=3)
    torch_output = torch.repeat_interleave(torch_output, scale_factor, dim=2)

    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(torch_tensor, device=device, dtype=ttnn.bfloat16)
    model = upsample_nearest2d(input_shape[-1], input_shape[-2], input_shape[-3], scale_factor)
    tt_out = model(input_tensor)
    tt_out = ttnn.from_device(tt_out)
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_out)
    tt_output = torch.permute(tt_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, tt_output, 0.9999)
