# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40), (2, 32, 128)])
@pytest.mark.parametrize("operation", ["topk", "max", "min", "sum", "mean"])
def test_reduction_preserves_input_bits(device, shape, operation):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    # Negative zero, a subnormal, and a NaN payload distinguish a raw copy from
    # a floating-point unpack/pack round trip through fill_pad.
    values.view(torch.int16).flatten()[:3] = torch.tensor([-32768, 1, 0x7FC1], dtype=torch.int16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before = ttnn.to_torch(tensor).view(torch.int16).clone()
    if operation == "topk":
        result, _ = ttnn.topk(tensor, k=8, dim=-1, largest=False)
    else:
        result = getattr(ttnn, operation)(tensor, dim=-1)
    ttnn.to_torch(result)
    after = ttnn.to_torch(tensor).view(torch.int16)
    assert torch.equal(after, before)


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40)])
@pytest.mark.parametrize("operation", ["topk", "max", "min", "sum", "mean"])
def test_reduction_with_private_padding_matches_reference(device, shape, operation):
    torch.manual_seed(0)
    values = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    if operation == "topk":
        result, _ = ttnn.topk(tensor, k=8, dim=-1, largest=False)
        expected = torch.topk(values, k=8, dim=-1, largest=False).values
    else:
        result = getattr(ttnn, operation)(tensor, dim=-1, keepdim=True)
        expected = getattr(torch, operation)(values.float(), dim=-1, keepdim=True)
        if operation in ("max", "min"):
            expected = expected.values
    torch.testing.assert_close(ttnn.to_torch(result).float(), expected.float(), rtol=0.02, atol=0.01)
