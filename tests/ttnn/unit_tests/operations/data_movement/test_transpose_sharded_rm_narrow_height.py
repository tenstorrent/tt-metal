# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device


def _height_sharded_on_one_core(shape, dtype, device):
    n, c, h, w = shape
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(dtype)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (n * c * h, w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config)
    return torch_input, tt_input


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((1, 4, 4, 64), torch.bfloat16),
        ((1, 4, 12, 64), torch.bfloat16),
        ((1, 4, 20, 64), torch.bfloat16),
        ((1, 4, 24, 64), torch.bfloat16),
        ((1, 4, 28, 64), torch.bfloat16),
        ((1, 4, 20, 64), torch.float32),
        ((1, 4, 36, 64), torch.bfloat16),
        ((1, 4, 48, 64), torch.bfloat16),
        ((1, 4, 100, 64), torch.bfloat16),
        ((1, 8, 16, 40), torch.bfloat16),
        ((1, 4, 8, 64), torch.bfloat16),
        ((1, 4, 16, 64), torch.bfloat16),
        ((1, 4, 4, 64), torch.float32),
        ((1, 4, 12, 64), torch.float32),
        ((1, 4, 16, 64), torch.float32),
        ((1, 4, 32, 64), torch.bfloat16),
    ],
)
def test_transpose_wh_height_sharded_rm_narrow_height(shape, dtype, device):
    torch_input, tt_input = _height_sharded_on_one_core(shape, dtype, device)

    tt_output = ttnn.transpose(tt_input, -2, -1)

    expected = torch_input.transpose(-2, -1)
    actual = ttnn.to_torch(tt_output)
    assert actual.shape == expected.shape
    mismatched = (actual != expected).sum().item()
    assert mismatched == 0, f"{mismatched} of {expected.numel()} datums wrong"
