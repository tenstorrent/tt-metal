# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0

pytestmark = pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only program")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "in_n, in_h, in_w, in_c, kernel, stride, pad",
    [
        (1, 128, 128, 128, 3, 1, 1),
        (1, 64, 64, 256, 3, 1, 1),
        (1, 80, 200, 16, 3, 2, 1),
        (2, 112, 112, 8, 3, 2, 1),
        (1, 57, 61, 32, 3, 1, 1),
        (1, 96, 96, 64, 5, 1, 2),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_max_pool2d_height_shard_exact(device, in_n, in_h, in_w, in_c, kernel, stride, pad, layout):
    torch.manual_seed(0)
    torch_input = torch.randn(in_n, in_c, in_h, in_w, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, in_n * in_h * in_w, in_c),
        ttnn.bfloat16,
        layout=layout,
        device=device,
    )
    padded = torch.nn.functional.pad(torch_input.float(), (pad, pad, pad, pad), value=-math.inf)
    expected = torch.nn.functional.max_pool2d(padded, kernel, stride)
    out_h, out_w = expected.shape[-2:]
    for _ in range(2):
        output = ttnn.max_pool2d(
            input_tensor=ttnn_input,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            kernel_size=[kernel, kernel],
            stride=[stride, stride],
            padding=[pad, pad],
            dilation=[1, 1],
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        output = ttnn.to_torch(output).float().reshape(in_n, out_h, out_w, in_c).permute(0, 3, 1, 2)
        assert torch.equal(output, expected)
