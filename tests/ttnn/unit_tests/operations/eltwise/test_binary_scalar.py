# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    (
        (ttnn.gt),
        (ttnn.lt),
        (ttnn.ne),
        (ttnn.ge),
        (ttnn.le),
        (ttnn.eq),
    ),
)
def test_binary_scalar_ops(input_shapes, device, ttnn_fn):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shapes, dtype=torch.bfloat16) * 100
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.zeros_like(input_tensor)
    scalar = random.randint(-80, 80)
    ttnn_fn(input_tensor, scalar, output_tensor=output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_fn(torch_input, scalar)

    out = ttnn.to_torch(output_tensor).to(torch.bool)

    assert torch.equal(out, golden_tensor)
